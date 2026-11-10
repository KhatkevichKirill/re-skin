"""Typed, idempotent knowledge writers and invalidation-aware readers.

Design: docs/knowledge-feedback-design.md §4 (schema semantics), §6.1 C2
(version allocation), §6.2 (lock hierarchy), §6.5 (scope), §8.2 (holdout).

PostgreSQL only — the services rely on row locks, advisory locks and
``ON CONFLICT``. Every writer runs inside the caller's transaction and never
commits; the caller commits or rolls back the whole request.

Lock hierarchy for every writer: idempotency-key advisory lock → ``runs`` row
(``FOR UPDATE``) → ``kb_cases`` rows in ascending id order. Nothing locks a case
and then a Run.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from sqlalchemy import exists, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from . import taxonomy as t
from .access import KnowledgeAccessDenied, KnowledgeScope
from .models import (
    KbArtifact, KbArtifactCheck, KbAttempt, KbAttemptObservation, KbAttemptSubmission, KbCaptureDone,
    KbCaptureQueue, KbCase, KbCaseVersion, KbDecision, KbDecisionOutcome, KbEvalManifest,
    KbEvalMembership, KbFeedbackEvent, KbImportBatch, KbImportBatchObservation, KbImportMember,
    KbInvalidation, KbRequest, KbRule, KbRuleEvidence, KbRuleReview, KbRuleVersion, KbSourceSubject,
    KbSubjectLink,
)

PRESERVED_LOCATORS = ("preserved_copy", "drive_file", "assistant_artifact")
TERMINAL_STATUSES = ("succeeded", "failed", "short_result", "refused", "superseded",
                     "intent_without_submission")
TIME_TOLERANCE_SEC = 0.05


class KnowledgeError(Exception):
    pass


class KnowledgeInvalid(KnowledgeError):
    """Request is malformed or violates a knowledge rule (422)."""


class KnowledgeConflict(KnowledgeError):
    """Stale version, reused idempotency key or concurrent conflicting state (409)."""

    def __init__(self, message: str, current_version_no: Optional[int] = None):
        super().__init__(message)
        self.current_version_no = current_version_no


class NotDelivered(KnowledgeConflict):
    """A Run case judged while its run is not ``done`` (S8): the final on disk
    may be mid-rewrite, so nothing is hashed or recorded."""


class BatchWithdrawn(KnowledgeConflict):
    """The import batch was invalidated mid-import: the importer must stop the pass
    without advancing, and continue under a new batch."""


# ---------------------------------------------------------------- helpers


def _require_pg(session: Session) -> None:
    if session.get_bind().dialect.name != "postgresql":
        raise KnowledgeError("the knowledge store requires PostgreSQL")


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _advisory_lock(session: Session, *parts, shared: bool = False) -> None:
    key = int.from_bytes(hashlib.sha256("\x00".join(map(str, parts)).encode()).digest()[:8], "big", signed=True)
    fn = "pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
    session.execute(text(f"SELECT {fn}(:k)"), {"k": key})


def _one_of(value, allowed, what: str):
    if value not in allowed:
        raise KnowledgeInvalid(f"unknown {what}: {value!r}")
    return value


def _finite_nonneg(value, what: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise KnowledgeInvalid(f"{what} must be a non-negative number")
    return value


def run_case_key(run_id: str) -> str:
    return f"run:{run_id}"


def segment_case_key(run_id: str, segment_def_id: str) -> str:
    return f"segment:{run_id}:{segment_def_id}"


def derived_case_key(table: str, row_id: str) -> str:
    return f"derived:{table}:{row_id}"


def manual_case_key(manual_creative_id: str) -> str:
    return f"manual:{manual_creative_id}"


def external_case_key(anchor: str, n: int) -> str:
    return f"external:{anchor}:{n}"


_KEY_PREFIX = {"run": "run:", "segment": "segment:", "derived": "derived:", "manual": "manual:",
               "external": "external:"}


# ---------------------------------------------------------------- invalidation


_INVALID_CACHE: dict = {}  # database URL -> (max invalidation seq, InvalidSet)


def withdrawn_clause(kind: str, id_column):
    """SQL predicate: the row named by ``id_column`` is withdrawn, directly or as a
    member of an invalidated import batch. Evaluated by indexed lookups
    (``uq_kb_invalidation_target``, ``uq_kb_import_member``), so a query's cost does
    not grow with the size of withdrawn batches."""
    direct = exists().where(KbInvalidation.target_kind == kind, KbInvalidation.target_id == id_column)
    via_batch = exists().where(
        KbImportMember.target_kind == kind, KbImportMember.target_id == id_column,
        exists().where(KbInvalidation.target_kind == "import_batch",
                       KbInvalidation.target_id == KbImportMember.batch_id))
    return direct | via_batch


def _batch_withdrawn(batch_id_column):
    return exists().where(KbInvalidation.target_kind == "import_batch", KbInvalidation.target_id == batch_id_column)


@dataclass
class InvalidSet:
    """Everything a read must exclude, directly or through its import batch."""
    ids: dict = field(default_factory=dict)  # kind -> set(ids)

    @classmethod
    def load(cls, session: Session) -> "InvalidSet":
        """Cached per process by the newest invalidation ``seq``: invalidations are
        append-only and an invalidated batch accepts no new members
        (``add_import_member``), so the set changes exactly when that number does.
        Steady state costs one indexed ``max(seq)`` query."""
        key = (session.get_bind().url.render_as_string(hide_password=True),
               session.execute(select(func.max(KbInvalidation.seq))).scalar())
        cached = _INVALID_CACHE.get(key[0])
        if cached is not None and cached[0] == key[1]:
            return cached[1]
        loaded = cls._load(session)
        _INVALID_CACHE[key[0]] = (key[1], loaded)
        return loaded

    @classmethod
    def _load(cls, session: Session) -> "InvalidSet":
        out: dict = {}
        rows = session.execute(select(KbInvalidation.target_kind, KbInvalidation.target_id)).all()
        batches = {tid for kind, tid in rows if kind == "import_batch"}
        for kind, tid in rows:
            out.setdefault(kind, set()).add(tid)
        if batches:
            for kind, tid in session.execute(select(KbImportMember.target_kind, KbImportMember.target_id)
                                             .where(KbImportMember.batch_id.in_(batches))).all():
                out.setdefault(kind, set()).add(tid)
        return cls(out)

    def has(self, kind: str, row_id) -> bool:
        return row_id is not None and row_id in self.ids.get(kind, ())

    @property
    def batches(self) -> set:
        return self.ids.get("import_batch", set())

    def row_invalid(self, kind: str, row) -> bool:
        return self.has(kind, row.id) or getattr(row, "import_batch_id", None) in self.batches

    def case_invalid(self, case: KbCase) -> bool:
        return self.row_invalid("case", case)

    def version_invalid(self, version: KbCaseVersion) -> bool:
        return self.has("case_version", version.id) or self.has("artifact", version.artifact_id)

    def event_invalid(self, event: KbFeedbackEvent) -> bool:
        return (self.row_invalid("event", event)
                or any(self.has("artifact", a) for a in (event.artifact_id, event.before_artifact_id,
                                                         event.after_artifact_id)))

    def attempt_invalid(self, attempt: KbAttempt) -> bool:
        return self.row_invalid("attempt", attempt)


# ---------------------------------------------------------------- import batches


def start_import_batch(session: Session, *, source: str, code_revision: str, dry_run: bool) -> KbImportBatch:
    _require_pg(session)
    batch = KbImportBatch(source=source, code_revision=code_revision, dry_run=dry_run)
    session.add(batch)
    session.flush()
    return batch


def finish_import_batch(session: Session, batch_id: str, *, kind: str, counts=None, report=None):
    _one_of(kind, t.IMPORT_OUTCOMES, "import outcome")
    session.add(KbImportBatchObservation(batch_id=batch_id, kind=kind, counts=counts, report=report))
    session.flush()


def add_import_member(session: Session, batch_id: str, target_kind: str, target_id: str) -> None:
    _one_of(target_kind, t.IMPORT_MEMBER_KINDS, "import member kind")
    # Shared lock vs invalidate()'s exclusive one: a batch is never invalidated while
    # a transaction is still adding members to it.
    _advisory_lock(session, "kb_import_batch", batch_id, shared=True)
    if session.execute(select(KbInvalidation.id).where(KbInvalidation.target_kind == "import_batch",
                                                       KbInvalidation.target_id == batch_id)).first():
        # Invariant behind InvalidSet's cache: a withdrawn batch never grows.
        raise BatchWithdrawn("this import batch was invalidated; start a new batch")
    session.execute(pg_insert(KbImportMember).values(id=str(uuid.uuid4()), batch_id=batch_id,
                                                     target_kind=target_kind, target_id=target_id)
                    .on_conflict_do_nothing(constraint="uq_kb_import_member"))


def invalidate(session: Session, *, target_kind: str, target_id: str, reason: str,
               actor_label: Optional[str] = None) -> KbInvalidation:
    """Append-only retraction of imported/derived evidence; idempotent per target."""
    _require_pg(session)
    _one_of(target_kind, t.INVALIDATION_TARGETS, "invalidation target")
    if not reason or not reason.strip():
        raise KnowledgeInvalid("an invalidation needs a reason")
    if target_kind == "import_batch":
        _advisory_lock(session, "kb_import_batch", target_id)  # waits for in-flight member writers
    session.execute(pg_insert(KbInvalidation).values(id=str(uuid.uuid4()), target_kind=target_kind,
                                                     target_id=target_id, reason=reason,
                                                     actor_label=actor_label)
                    .on_conflict_do_nothing(constraint="uq_kb_invalidation_target"))
    return session.execute(select(KbInvalidation).where(KbInvalidation.target_kind == target_kind,
                                                        KbInvalidation.target_id == target_id)).scalar_one()


# ---------------------------------------------------------------- artifacts


def get_or_create_artifact(session: Session, *, kind: str, locator_kind: str, locator: str,
                           sha256: Optional[str], project_id: Optional[str], bytes: Optional[int] = None,
                           duration_sec: Optional[float] = None, width: Optional[int] = None,
                           height: Optional[int] = None, parent_artifact_id: Optional[str] = None,
                           time_start_sec: Optional[float] = None,
                           time_end_sec: Optional[float] = None) -> KbArtifact:
    """Content-addressed when hashed: the same locator + bytes is one artifact.

    An unhashed artifact (``sha256`` NULL = unknown content) is always a new row:
    two unknowns at one path need not be the same content.
    """
    _require_pg(session)
    _one_of(kind, t.ARTIFACT_KINDS, "artifact kind")
    _one_of(locator_kind, t.LOCATOR_KINDS, "locator kind")
    if not locator:
        raise KnowledgeInvalid("artifact locator is required")
    if sha256 is not None and (len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256)):
        raise KnowledgeInvalid("sha256 must be 64 lowercase hex characters")
    _finite_nonneg(duration_sec, "duration_sec")
    _validate_range(time_start_sec, time_end_sec, None)
    values = dict(kind=kind, locator_kind=locator_kind, locator=locator, sha256=sha256, project_id=project_id,
                  bytes=bytes, duration_sec=duration_sec, width=width, height=height,
                  parent_artifact_id=parent_artifact_id, time_start_sec=time_start_sec,
                  time_end_sec=time_end_sec)
    if sha256 is None:
        artifact = KbArtifact(**values)
        session.add(artifact)
        session.flush()
        return artifact
    same = (KbArtifact.locator_kind == locator_kind, KbArtifact.locator == locator, KbArtifact.sha256 == sha256)
    artifact = session.execute(select(KbArtifact).where(*same, ~withdrawn_clause("artifact", KbArtifact.id))
                               .order_by(KbArtifact.revision.desc()).limit(1)).scalar_one_or_none()
    if artifact is None:  # first sighting, or every earlier record was withdrawn: a new revision
        revision = (session.execute(select(func.max(KbArtifact.revision)).where(*same)).scalar() or 0) + 1
        session.execute(pg_insert(KbArtifact).values(id=str(uuid.uuid4()), revision=revision, **values)
                        .on_conflict_do_nothing(constraint="uq_kb_artifact_locator_content"))
        artifact = session.execute(select(KbArtifact).where(*same, KbArtifact.revision == revision)).scalar_one()
    if artifact.project_id != project_id:
        raise KnowledgeConflict("artifact already recorded for a different project")
    return artifact


def record_artifact_check(session: Session, artifact_id: str, *, present: bool,
                          sha256_matches: Optional[bool]) -> KbArtifactCheck:
    check = KbArtifactCheck(artifact_id=artifact_id, present=present, sha256_matches=sha256_matches)
    session.add(check)
    session.flush()
    return check


# ---------------------------------------------------------------- cases


_CASE_IDENTITY = ("scope", "operation", "project_id", "run_id", "run_segment_id", "segment_def_id",
                  "parent_case_id")


def ensure_case(session: Session, *, case_key: str, scope: str, operation: str, project_id: Optional[str],
                origin: str = "live", import_batch_id: Optional[str] = None, run_id: Optional[str] = None,
                run_segment_id: Optional[str] = None, segment_def_id: Optional[str] = None,
                parent_case_id: Optional[str] = None, source_artifact_id: Optional[str] = None,
                source_group_key: Optional[str] = None, language: Optional[str] = None,
                time_start_sec: Optional[float] = None, time_end_sec: Optional[float] = None,
                snapshot: Optional[dict] = None, missing_facts: Optional[list] = None) -> KbCase:
    """Insert-or-get by ``case_key``; a key bound to different identity is a conflict.

    Callers creating a Run-bound case hold the Run lock (§6.2) so the case row
    they then lock is never created behind another writer's back.
    """
    _require_pg(session)
    _one_of(scope, t.CASE_SCOPES, "case scope")
    _one_of(operation, t.CASE_OPERATIONS, "operation")
    _one_of(origin, t.ORIGINS, "origin")
    if not case_key.startswith(_KEY_PREFIX[scope]):
        raise KnowledgeInvalid(f"case key {case_key!r} does not match scope {scope}")
    if (origin == "import") != (import_batch_id is not None):
        raise KnowledgeInvalid("imported cases carry their batch and only they do")
    if project_id is None and scope != "external":
        raise KnowledgeInvalid("only an external case may lack a project")
    _validate_range(time_start_sec, time_end_sec, None)
    invalid = InvalidSet.load(session)
    if parent_case_id is not None:
        parent = session.get(KbCase, parent_case_id)
        if parent is None or parent.project_id != project_id or invalid.case_invalid(parent):
            raise KnowledgeInvalid("parent case must be a current case in the same project")
    values = dict(case_key=case_key, scope=scope, operation=operation, project_id=project_id, run_id=run_id,
                  run_segment_id=run_segment_id, segment_def_id=segment_def_id, parent_case_id=parent_case_id,
                  source_artifact_id=source_artifact_id, source_group_key=source_group_key, language=language,
                  time_start_sec=time_start_sec, time_end_sec=time_end_sec, origin=origin,
                  import_batch_id=import_batch_id, snapshot=snapshot or {}, missing_facts=missing_facts or [])
    case = current_case(session, case_key, invalid)
    if case is None:  # first sighting, or every earlier revision was withdrawn
        last = session.execute(select(func.max(KbCase.revision)).where(KbCase.case_key == case_key)).scalar()
        revision = (last or 0) + 1
        session.execute(pg_insert(KbCase).values(id=str(uuid.uuid4()), revision=revision, **values)
                        .on_conflict_do_nothing(constraint="uq_kb_case_key"))
        case = session.execute(select(KbCase).where(KbCase.case_key == case_key,
                                                    KbCase.revision == revision)).scalar_one()
    for name in _CASE_IDENTITY:
        if getattr(case, name) != values[name]:
            raise KnowledgeConflict(f"case {case_key} already exists with a different {name}")
    return case


def current_case(session: Session, case_key: str, invalid: Optional[InvalidSet] = None) -> Optional[KbCase]:
    """Highest non-withdrawn revision for ``case_key`` (None if none)."""
    # One indexed row: withdrawn revisions (directly, as batch members, or through
    # their own import batch) are excluded in SQL, never loaded (runs under Run locks).
    return session.execute(select(KbCase).where(
        KbCase.case_key == case_key, ~withdrawn_clause("case", KbCase.id), ~_batch_withdrawn(KbCase.import_batch_id))
        .order_by(KbCase.revision.desc()).limit(1)).scalar_one_or_none()


def lock_run_and_cases(session: Session, run_id: Optional[str], case_ids: Iterable[str]) -> list[KbCase]:
    """§6.2 hierarchy: the Run row first (if it still exists), then cases by id."""
    from ..models import Run

    if run_id is not None:
        session.execute(select(Run.id).where(Run.id == run_id).with_for_update())
    ids = sorted(set(case_ids))
    if not ids:
        return []
    rows = session.execute(select(KbCase).where(KbCase.id.in_(ids)).order_by(KbCase.id)
                           .with_for_update()).scalars().all()
    return rows


def versions(session: Session, case_id: str, invalid: Optional[InvalidSet] = None, *,
             current_only: bool = False) -> list[KbCaseVersion]:
    """All valid versions by number; ``current_only`` drops historical ones."""
    rows = session.execute(select(KbCaseVersion).where(KbCaseVersion.case_id == case_id)
                           .order_by(KbCaseVersion.version_no)).scalars().all()
    return [v for v in rows if (invalid is None or not invalid.version_invalid(v))
            and not (current_only and v.historical)]


def latest_version(session: Session, case_id: str, invalid: Optional[InvalidSet] = None):
    """The current version: the highest valid, non-historical one (one indexed
    row; withdrawn versions/artifacts are excluded in SQL, never by loading
    the timeline — this runs under Run locks)."""
    invalid = invalid if invalid is not None else InvalidSet.load(session)
    q = select(KbCaseVersion).where(KbCaseVersion.case_id == case_id, KbCaseVersion.historical.is_(False))
    q = q.where(~withdrawn_clause("case_version", KbCaseVersion.id),
                ~withdrawn_clause("artifact", KbCaseVersion.artifact_id))
    return session.execute(q.order_by(KbCaseVersion.version_no.desc()).limit(1)).scalar_one_or_none()


SOURCE_RECORD_MAX = 180  # leaves room for a "~<n>" re-record suffix in the 200-character column


def _check_record(record) -> None:
    """Caller-supplied publication ids: 1-180 characters, no "~" (reserved for re-records)."""
    if not isinstance(record, str) or not 1 <= len(record) <= SOURCE_RECORD_MAX or "~" in record:
        raise KnowledgeInvalid(f"a publication id is 1-{SOURCE_RECORD_MAX} characters without '~'")


def _publication_filter(case_id: str, source_record: str):
    """Exactly ``<record>`` or ``<record>~<digits>``, decided in SQL (index-friendly)."""
    return (KbCaseVersion.case_id == case_id) & (
        (KbCaseVersion.source_record == source_record)
        | KbCaseVersion.source_record.op("~")("^" + re.escape(source_record) + "~[0-9]+$"))


def publication_versions(session: Session, case_id: str, source_record: str) -> list[KbCaseVersion]:
    """Every version recorded for one publication (all revisions; prefer the
    bounded helpers below on hot paths)."""
    return session.execute(select(KbCaseVersion).where(_publication_filter(case_id, source_record))
                           .order_by(KbCaseVersion.version_no)).scalars().all()


def valid_publication_version(session: Session, case_id: str, source_record: str,
                              invalid: "InvalidSet") -> Optional[KbCaseVersion]:
    """The newest non-withdrawn version of a publication — one indexed row, with
    withdrawn versions/artifacts excluded in SQL, never the revision history."""
    q = select(KbCaseVersion).where(_publication_filter(case_id, source_record))
    q = q.where(~withdrawn_clause("case_version", KbCaseVersion.id),
                ~withdrawn_clause("artifact", KbCaseVersion.artifact_id))
    return session.execute(q.order_by(KbCaseVersion.version_no.desc()).limit(1)).scalar_one_or_none()


def publication_record_count(session: Session, case_id: str, source_record: str) -> int:
    return session.execute(select(func.count()).select_from(KbCaseVersion)
                           .where(_publication_filter(case_id, source_record))).scalar()


def append_case_version(session: Session, case: KbCase, *, artifact_id: str, reason: str,
                        source_record: Optional[str] = None, historical: bool = False,
                        published_at: Optional[datetime] = None,
                        invalid: Optional[InvalidSet] = None) -> KbCaseVersion:
    """Caller holds the case row lock (and the Run lock for a Run case).

    ``source_record`` names the publication and is unique per case: a retry of
    the same publication returns the existing version. A ``historical`` version
    takes the next number but never becomes current.
    """
    _one_of(reason, t.VERSION_REASONS, "version reason")
    if historical and case.scope in ("derived", "manual", "external"):
        raise KnowledgeInvalid(f"a {case.scope} case has no history beyond its one version")
    if source_record is not None:
        # The record is the publication, once; a record whose version was
        # withdrawn may be re-recorded under "<record>~<n>".
        _check_record(source_record)
        invalid = invalid if invalid is not None else InvalidSet.load(session)  # callers in a pass pass theirs
        existing = valid_publication_version(session, case.id, source_record, invalid)
        if existing is not None:
            if existing.artifact_id != artifact_id or existing.historical != bool(historical):
                raise KnowledgeConflict("this publication is already recorded with different content",
                                        existing.version_no)
            return existing
        prior = publication_record_count(session, case.id, source_record)
        if prior:
            source_record = f"{source_record}~{prior}"
    artifact = session.get(KbArtifact, artifact_id)
    if artifact is None:
        raise KnowledgeInvalid("unknown artifact")
    if case.project_id is not None and artifact.project_id != case.project_id:
        raise KnowledgeInvalid("artifact belongs to a different project")
    current = session.execute(select(func.max(KbCaseVersion.version_no))
                              .where(KbCaseVersion.case_id == case.id)).scalar()
    if current and case.scope in ("derived", "manual", "external"):
        raise KnowledgeConflict(f"a {case.scope} case has exactly one version", current)
    version = KbCaseVersion(case_id=case.id, version_no=(current or 0) + 1, artifact_id=artifact_id,
                            reason=reason, source_record=source_record, historical=bool(historical),
                            published_at=published_at)
    session.add(version)
    session.flush()
    return version


SYNC_DRAIN_MAX = 20


def sync_run_case_version(session: Session, case: KbCase, *, hasher: Callable[[str], str] = sha256_file,
                          invalid: Optional[InvalidSet] = None,
                          live_link_lookup: Optional[Callable] = None) -> Optional[KbCaseVersion]:
    """§6.2 step 3: append a version when the live Run final changed.

    Caller holds the Run lock, so ``result_local_path`` cannot move meanwhile.
    """
    from ..models import Run

    if case.scope != "run" or case.run_id is None:
        return None
    # Queued finals are older publications than the live file: allocate them
    # first, or a later drain would append one of them over the synced version.
    drain_run_captures(session, case.run_id, max_items=SYNC_DRAIN_MAX)
    if session.execute(select(KbCaptureQueue.id).where(
            KbCaptureQueue.run_id == case.run_id,
            ~exists().where(KbCaptureDone.queue_id == KbCaptureQueue.id))).first():
        # A backlog stays with the knowledge worker; never hold the Run row for it here.
        raise KnowledgeConflict("this run's captures are still being recorded; retry shortly")
    invalid = InvalidSet.load(session)
    path = _locked_result_path(session, case.run_id)
    if not path or not os.path.isfile(path):
        return None
    sha = hasher(path)
    latest = latest_version(session, case.id, invalid)
    # AICP-299 S14: when capture preserved the live final as a publication link
    # (same inode), record THAT immutable copy under its publication id, so the
    # scanner's later capture of the link is the same publication, not a second
    # unpreserved version.
    link = (live_link_lookup if live_link_lookup is not None else _default_live_link_lookup)(
        case.project_id, case.run_id, path)
    if link is not None:
        if valid_publication_version(session, case.id, link.publication_id, invalid) is not None:
            return None
        if latest is not None and session.get(KbArtifact, latest.artifact_id).sha256 == sha:
            return None  # same bytes: their preserved copy covers this version (bytes_preserved)
        from .publication import REASON_FOR_KIND

        artifact = get_or_create_artifact(session, kind="delivered_final", locator_kind="preserved_copy",
                                          locator=link.path, sha256=sha, project_id=case.project_id,
                                          bytes=os.path.getsize(path))
        reason = REASON_FOR_KIND.get(link.kind, "restitch")
        return append_case_version(session, case, artifact_id=artifact.id,
                                   reason=reason if latest is not None or reason != "restitch" else "initial_output",
                                   source_record=link.publication_id, invalid=invalid)
    if latest is not None and session.get(KbArtifact, latest.artifact_id).sha256 == sha:
        return None
    artifact = get_or_create_artifact(session, kind="delivered_final", locator_kind="local_path", locator=path,
                                      sha256=sha, project_id=case.project_id, bytes=os.path.getsize(path))
    next_no = (session.execute(select(func.max(KbCaseVersion.version_no))
                               .where(KbCaseVersion.case_id == case.id)).scalar() or 0) + 1
    return append_case_version(session, case, artifact_id=artifact.id,
                               reason="restitch" if latest is not None else "initial_output",
                               source_record=f"sync:{case.run_id}:{sha}:{next_no}", invalid=invalid)


def bytes_preserved(session: Session, artifact: KbArtifact) -> bool:
    """S10, content-addressed: are this artifact's BYTES kept immutably somewhere?

    True for a preserved locator, and for a hashed mutable ``local_path`` whose
    exact bytes (same sha256, same project) also exist as a valid preserved
    artifact. That is how Start can record the live final before capture links
    it: the drain then drops the link as a same-bytes duplicate, so feedback
    stays on the version it was given for, and this check still sees the bytes
    as preserved. Withdrawn preserved copies do not count.
    """
    if artifact.locator_kind in PRESERVED_LOCATORS:
        return True
    if artifact.sha256 is None:
        return False
    return session.execute(select(KbArtifact.id).where(
        KbArtifact.sha256 == artifact.sha256,
        KbArtifact.project_id.is_not_distinct_from(artifact.project_id),
        KbArtifact.locator_kind.in_(PRESERVED_LOCATORS),
        ~withdrawn_clause("artifact", KbArtifact.id)).limit(1)).first() is not None


def _default_live_link_lookup(project_id, run_id, live_path):
    """Filesystem-only lookup of the live final's publication link (links.py)."""
    try:
        from .links import live_link_lookup
    except Exception:  # pragma: no cover
        return None
    return live_link_lookup(project_id, run_id, live_path)


# ---------------------------------------------------------------- attempts


def reference_set_sha256(reference_hashes) -> Optional[str]:
    """Ordered reference content hashes → one id; unknown if any is unknown."""
    if not reference_hashes or any(not h for h in reference_hashes):
        return None
    return canonical_sha256(list(reference_hashes))


def dispatch_key(kind: str, target_kind, target_id, predecessor_task_id, request_sha256) -> str:
    return canonical_sha256([kind, target_kind, target_id, predecessor_task_id or "none", request_sha256])


def record_attempt_intent(session: Session, *, case_id: str, attempt_kind: str, origin: str, provider: str,
                          request: Optional[dict], target_kind: Optional[str] = None,
                          target_id: Optional[str] = None, predecessor_task_id: Optional[str] = None,
                          model_key: Optional[str] = None, provider_model_id: Optional[str] = None,
                          model_family: Optional[str] = None, input_hashes: Optional[dict] = None,
                          recipe_context: Optional[dict] = None, code_revision: Optional[str] = None,
                          parent_attempt_id: Optional[str] = None, idempotency_key: Optional[str] = None,
                          import_batch_id: Optional[str] = None,
                          redispatch_payload: Optional[dict] = None,
                          dispatch_basis: Optional[dict] = None) -> tuple[KbAttempt, bool]:
    """§6.1 A — durable intent before a provider call. Returns (attempt, reused).

    Under an advisory lock on the target, an *open* intent (no submission, no
    terminal status) with the same dispatch key is reused and gets a
    ``redispatch`` observation: a recovered job resending the same recipe is the
    same logical dispatch, flagged as possibly charged twice.
    """
    _require_pg(session)
    _one_of(attempt_kind, t.ATTEMPT_KINDS, "attempt kind")
    _one_of(origin, t.ATTEMPT_ORIGINS, "attempt origin")
    _one_of(provider, t.PROVIDERS, "provider")
    if session.get(KbCase, case_id) is None:
        raise KnowledgeInvalid("unknown case")
    request_sha = canonical_sha256(request) if request is not None else None
    # AICP-299 S1: a recovered job re-uploads its references (new URLs), so a
    # live capture keys the dispatch on its locators (``dispatch_basis``), not
    # on the volatile request bytes.
    dkey = dispatch_key(attempt_kind, target_kind, target_id, predecessor_task_id,
                        canonical_sha256(dispatch_basis) if dispatch_basis is not None else request_sha)
    identity = dict(case_id=case_id, attempt_kind=attempt_kind, origin=origin, provider=provider,
                    model_key=model_key, provider_model_id=provider_model_id, model_family=model_family,
                    parent_attempt_id=parent_attempt_id, input_hashes=input_hashes,
                    recipe_context=recipe_context, code_revision=code_revision, import_batch_id=import_batch_id)
    if idempotency_key is not None:
        _advisory_lock(session, "kb_attempt_key", idempotency_key)
        existing = session.execute(select(KbAttempt).where(KbAttempt.idempotency_key == idempotency_key)
                                   ).scalar_one_or_none()
        if existing is not None:
            if existing.dispatch_key != dkey or any(getattr(existing, k) != v for k, v in identity.items()):
                raise KnowledgeConflict("attempt idempotency key reused for a different dispatch")
            if InvalidSet.load(session).attempt_invalid(existing):
                # Keys are unique forever; the caller records the dispatch under a new key.
                raise KnowledgeConflict("this attempt was withdrawn; record the dispatch under a new key")
            return existing, True
    if target_kind is not None:
        _advisory_lock(session, "kb_attempt_target", target_kind, target_id)
        invalid = InvalidSet.load(session)
        for candidate in session.execute(select(KbAttempt).where(KbAttempt.dispatch_key == dkey)
                                         .order_by(KbAttempt.seq.desc())).scalars():
            # Same logical dispatch only: same provider/model provenance, still valid, still open.
            if invalid.attempt_invalid(candidate) or any(
                    getattr(candidate, k) != identity[k] for k in ("case_id", "provider", "model_key",
                                                                  "provider_model_id")):
                continue
            if _attempt_is_open(session, candidate.id, invalid):
                record_observation(session, candidate.id, kind="redispatch", payload=redispatch_payload or {})
                return candidate, True
            break  # the newest valid match is closed: a fresh dispatch
    refs = (input_hashes or {}).get("references")
    attempt = KbAttempt(case_id=case_id, idempotency_key=idempotency_key or str(uuid.uuid4()),
                        attempt_kind=attempt_kind, origin=origin, parent_attempt_id=parent_attempt_id,
                        provider=provider, model_key=model_key, provider_model_id=provider_model_id,
                        model_family=model_family, target_kind=target_kind, target_id=target_id,
                        dispatch_key=dkey, predecessor_task_id=predecessor_task_id, request=request,
                        request_sha256=request_sha, input_hashes=input_hashes,
                        reference_set_sha256=reference_set_sha256(refs) if refs is not None else None,
                        recipe_context=recipe_context, code_revision=code_revision,
                        import_batch_id=import_batch_id)
    session.add(attempt)
    session.flush()
    return attempt, False


def _attempt_is_open(session: Session, attempt_id: str, invalid: InvalidSet) -> bool:
    """No valid submission and no valid terminal status observation."""
    sub = session.execute(select(KbAttemptSubmission).where(KbAttemptSubmission.attempt_id == attempt_id)
                          ).scalar_one_or_none()
    if sub is not None and not invalid.has("submission", sub.id):
        return False
    for obs in session.execute(select(KbAttemptObservation).where(
            KbAttemptObservation.attempt_id == attempt_id, KbAttemptObservation.kind == "status",
            KbAttemptObservation.status.in_(TERMINAL_STATUSES))).scalars():
        if not invalid.row_invalid("observation", obs):
            return False
    return True


def record_submission(session: Session, attempt_id: str, *, provider_task_id: str,
                      spend_call_key: Optional[str] = None) -> KbAttemptSubmission:
    """Provider acceptance + ``status=submitted`` in the caller's one transaction."""
    _require_pg(session)
    if not provider_task_id:
        raise KnowledgeInvalid("provider task id is required")
    _advisory_lock(session, "kb_attempt_submission", attempt_id)  # replay is read-then-insert
    existing = session.execute(select(KbAttemptSubmission).where(KbAttemptSubmission.attempt_id == attempt_id)
                               ).scalar_one_or_none()
    if existing is not None:
        if existing.provider_task_id != provider_task_id or existing.spend_call_key != spend_call_key:
            raise KnowledgeConflict("attempt already has a different submission")
        return existing
    sub = KbAttemptSubmission(attempt_id=attempt_id, provider_task_id=provider_task_id,
                              spend_call_key=spend_call_key)
    session.add(sub)
    session.flush()
    record_observation(session, attempt_id, kind="status", status="submitted")
    return sub


def record_observation(session: Session, attempt_id: str, *, kind: str, status: Optional[str] = None,
                       output_artifact_id: Optional[str] = None, mcredits: Optional[int] = None,
                       cost_source: Optional[str] = None, error_class: Optional[str] = None,
                       payload: Optional[dict] = None,
                       import_batch_id: Optional[str] = None) -> KbAttemptObservation:
    _one_of(kind, t.OBSERVATION_KINDS, "observation kind")
    if status is not None:
        _one_of(status, t.ATTEMPT_STATUSES, "attempt status")
    if cost_source is not None:
        _one_of(cost_source, t.COST_SOURCES, "cost source")
    if kind == "status" and status is None:
        raise KnowledgeInvalid("a status observation needs a status")
    if kind in ("cost_estimate", "cost_settled") and (mcredits is None or cost_source is None):
        raise KnowledgeInvalid("a cost observation needs mcredits and its source")
    if kind == "output":
        if output_artifact_id is None:
            raise KnowledgeInvalid("an output observation needs its artifact")
        attempt, artifact = session.get(KbAttempt, attempt_id), session.get(KbArtifact, output_artifact_id)
        case = session.get(KbCase, attempt.case_id) if attempt else None
        if artifact is None or case is None or (case.project_id is not None
                                                and artifact.project_id != case.project_id):
            raise KnowledgeInvalid("output artifact must belong to the attempt's project")
    obs = KbAttemptObservation(attempt_id=attempt_id, kind=kind, status=status,
                               output_artifact_id=output_artifact_id, mcredits=mcredits,
                               cost_source=cost_source, error_class=error_class, payload=payload,
                               import_batch_id=import_batch_id)
    session.add(obs)
    session.flush()
    return obs


def attempt_state(session: Session, attempt_id: str, invalid: Optional[InvalidSet] = None) -> dict:
    """Latest valid observation of each kind (by ``seq``) plus derived flags."""
    invalid = invalid if invalid is not None else InvalidSet.load(session)
    attempt = session.get(KbAttempt, attempt_id)
    if attempt is None or invalid.attempt_invalid(attempt):
        return {"status": None, "provider_task_id": None, "output_artifact_id": None, "estimated_mcredits": None,
                "settled_mcredits": None, "cost_source": None, "possibly_charged_more_than_once": False,
                "observations": [], "invalidated": True}
    rows = [o for o in session.execute(select(KbAttemptObservation)
                                       .where(KbAttemptObservation.attempt_id == attempt_id)
                                       .order_by(KbAttemptObservation.seq)).scalars()
            if not invalid.row_invalid("observation", o)
            and not (o.output_artifact_id and invalid.has("artifact", o.output_artifact_id))]
    latest = {}
    for obs in rows:
        latest[obs.kind] = obs
    sub = session.execute(select(KbAttemptSubmission).where(KbAttemptSubmission.attempt_id == attempt_id)
                          ).scalar_one_or_none()
    if sub is not None and invalid.has("submission", sub.id):
        sub = None
    return {
        "status": latest["status"].status if "status" in latest else None,
        "provider_task_id": sub.provider_task_id if sub else None,
        "output_artifact_id": latest["output"].output_artifact_id if "output" in latest else None,
        "estimated_mcredits": latest["cost_estimate"].mcredits if "cost_estimate" in latest else None,
        "settled_mcredits": latest["cost_settled"].mcredits if "cost_settled" in latest else None,
        "cost_source": next((latest[k].cost_source for k in ("cost_settled", "cost_estimate") if k in latest),
                            None),
        "possibly_charged_more_than_once": any(o.kind == "redispatch" for o in rows),
        "observations": rows,
        "invalidated": False,
    }


# ---------------------------------------------------------------- requests / receipts


@dataclass(frozen=True)
class Receipt:
    request_id: str
    kind: str
    case_id: Optional[str]
    ids: tuple
    replayed: bool


def _check_key(key) -> str:
    if not isinstance(key, str) or not 1 <= len(key) <= 128:
        raise KnowledgeInvalid("idempotency_key must be 1-128 characters")
    return key


def _replay(session: Session, key: str, request_sha: str) -> Optional[Receipt]:
    """Take the key lock; return the stored receipt for a replay, 409 on reuse."""
    _advisory_lock(session, "kb_request", key)
    existing = session.get(KbRequest, key)
    if existing is None:
        return None
    if existing.request_sha256 != request_sha:
        raise KnowledgeConflict("idempotency key already used for a different request")
    return Receipt(existing.idempotency_key, existing.kind, existing.case_id, tuple(existing.child_event_ids), True)


def _store_request(session: Session, key, request_sha, kind, case_id, ids) -> Receipt:
    session.add(KbRequest(idempotency_key=key, request_sha256=request_sha, kind=kind, case_id=case_id,
                          child_event_ids=list(ids)))
    session.flush()
    return Receipt(key, kind, case_id, tuple(ids), False)


# ---------------------------------------------------------------- case state


def _scoped_case(session: Session, scope: KnowledgeScope, case_id: str, invalid: InvalidSet) -> KbCase:
    case = session.get(KbCase, case_id)
    if case is None or invalid.case_invalid(case):
        raise KnowledgeAccessDenied("knowledge resource unavailable")
    scope.require_project(case.project_id)
    return case


def effective_events(session: Session, case_id: str, invalid: Optional[InvalidSet] = None) -> list:
    """Non-retraction events with no valid retraction pointing at them (§4.5)."""
    invalid = invalid if invalid is not None else InvalidSet.load(session)
    rows = session.execute(select(KbFeedbackEvent).where(KbFeedbackEvent.case_id == case_id)
                           .order_by(KbFeedbackEvent.seq)).scalars().all()
    # A withdrawn case version takes every judgement of it along (retroactively).
    withdrawn_nos = {v.version_no for v in session.execute(select(KbCaseVersion).where(
        KbCaseVersion.case_id == case_id)).scalars() if invalid.version_invalid(v)}
    valid = [e for e in rows if not invalid.event_invalid(e)
             and (e.case_version_no is None or e.case_version_no not in withdrawn_nos)]
    retracted = {e.supersedes_event_id for e in valid if e.event_type == "retraction"}
    return [e for e in valid if e.event_type != "retraction" and e.id not in retracted]


def _verdict_for(events, version_no) -> Optional[str]:
    picked = None
    for e in events:  # ordered by seq
        if e.event_type == "operator_verdict" and not e.historical and e.case_version_no == version_no:
            picked = e.verdict
    return picked


def case_state(session: Session, case_id: str, invalid: Optional[InvalidSet] = None) -> dict:
    """Deterministic verdict state (§4.5): only the latest version can be accepted."""
    invalid = invalid if invalid is not None else InvalidSet.load(session)
    vs = versions(session, case_id, invalid, current_only=True)
    events = effective_events(session, case_id, invalid)
    if not vs:
        return {"latest_version_no": None, "verdict": "unreviewed", "accepted_artifact_id": None,
                "previous_accepted_version_no": None}
    latest = vs[-1]
    verdict = _verdict_for(events, latest.version_no) or "unreviewed"
    previous = next((v.version_no for v in reversed(vs[:-1])
                     if _verdict_for(events, v.version_no) == "accepted_as_is"), None)
    return {"latest_version_no": latest.version_no, "latest_artifact_id": latest.artifact_id,
            "verdict": verdict,
            "accepted_artifact_id": latest.artifact_id if verdict == "accepted_as_is" else None,
            "previous_accepted_version_no": previous}


def _validate_range(start, end, duration) -> None:
    if start is None and end is None:
        return
    if start is None or end is None:
        raise KnowledgeInvalid("a time range needs both start and end")
    _finite_nonneg(start, "time_start_sec")
    _finite_nonneg(end, "time_end_sec")
    if end <= start:
        raise KnowledgeInvalid("time range end must be after its start")
    if duration is not None and end > duration + TIME_TOLERANCE_SEC:
        raise KnowledgeInvalid("time range lies outside the judged artifact")


def _run_is_done(session: Session, run_id: str) -> bool:
    """Fresh read under the caller's Run lock (never the identity map)."""
    from ..models import Run

    status = session.execute(select(Run.status).where(Run.id == run_id)
                             .execution_options(populate_existing=True)).scalar()
    return getattr(status, "value", status) == "done"


def _locked_case_and_version(session, scope, case_id, invalid, expected_version_no, historical, sync, hasher,
                             require_done=False):
    case = _scoped_case(session, scope, case_id, invalid)
    lock_run_and_cases(session, case.run_id if case.scope == "run" else None, [case.id])
    if require_done and case.scope == "run" and not _run_is_done(session, case.run_id):
        raise NotDelivered("not delivered: this run's output can be judged once the run is done")
    if sync and case.scope == "run":
        sync_run_case_version(session, case, hasher=hasher, invalid=invalid)
    vs = versions(session, case.id, invalid)
    current = [v for v in vs if not v.historical]
    if not current:
        raise KnowledgeConflict("this case has no recorded output yet")
    latest = current[-1]
    if not isinstance(expected_version_no, int) or isinstance(expected_version_no, bool):
        raise KnowledgeInvalid("expected_version_no is required")
    if historical:
        target = next((v for v in vs if v.version_no == expected_version_no), None)
        if target is None:
            raise KnowledgeInvalid("unknown version for a historical judgement")
        if target.version_no == latest.version_no:
            raise KnowledgeInvalid("the latest version is current, not historical")
        if not bytes_preserved(session, session.get(KbArtifact, target.artifact_id)):
            # S10: a mutable path may hold other bytes now; only preserved
            # history can be judged after the fact.
            raise KnowledgeInvalid("this version's bytes were not preserved; it cannot be judged historically")
    elif expected_version_no != latest.version_no:
        raise KnowledgeConflict("the output changed since this page was loaded", latest.version_no)
    else:
        target = latest
    return case, target


# ---------------------------------------------------------------- feedback


def submit_feedback(session: Session, scope: KnowledgeScope, *, case_id: str, idempotency_key: str,
                    expected_version_no: int, verdict: Optional[str] = None, defects: Iterable[dict] = (),
                    note: Optional[str] = None, historical: bool = False, actor_label: Optional[str] = None,
                    sync: bool = True, hasher: Callable[[str], str] = sha256_file,
                    require_done: bool = False) -> Receipt:
    """One operator submit = one receipt + its verdict/defect events, atomically (§4.5, §6.2).

    ``require_done`` (S8, the feedback API passes it): a Run case is judged
    only while its run is ``done``, checked under the Run lock before sync."""
    _require_pg(session)
    key = _check_key(idempotency_key)
    defects = [dict(d) for d in defects]
    body = {"case_id": case_id, "expected_version_no": expected_version_no, "verdict": verdict,
            "defects": defects, "note": note, "historical": bool(historical), "actor_label": actor_label}
    request_sha = canonical_sha256(["feedback", body])
    invalid = InvalidSet.load(session)
    # Scope first, so a replay cannot reveal a receipt for an out-of-scope case.
    _scoped_case(session, scope, case_id, invalid)
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return replay
    if verdict is None and not defects:
        raise KnowledgeInvalid("nothing to record: give a verdict or at least one defect")
    if verdict is not None:
        _one_of(verdict, t.VERDICTS, "verdict")
    case, target = _locked_case_and_version(session, scope, case_id, invalid, expected_version_no,
                                            bool(historical), sync, hasher, require_done)
    artifact = session.get(KbArtifact, target.artifact_id)
    for d in defects:
        _one_of(d.get("code"), t.DEFECT_CODES, "defect code")
        _one_of(d.get("severity"), t.SEVERITIES, "severity")
        _validate_range(d.get("start"), d.get("end"), artifact.duration_sec)
        if d["code"] == "other" and not (d.get("note") or "").strip():
            raise KnowledgeInvalid("defect 'other' needs a note")
    if not historical:
        current = case_state(session, case.id, invalid)["verdict"]
        serious = any(d["severity"] in ("blocker", "major") for d in defects)
        if serious and (current if verdict is None else verdict) == "accepted_as_is":
            raise KnowledgeConflict("a blocker/major defect contradicts the acceptance; "
                                    "submit Needs repair or Reject with it", target.version_no)
        if verdict == "accepted_as_is" and any(
                e.event_type == "defect" and not e.historical and e.case_version_no == target.version_no
                and e.severity in ("blocker", "major") for e in effective_events(session, case.id, invalid)):
            raise KnowledgeConflict("this version has a live blocker/major defect; retract it before accepting",
                                    target.version_no)
    common = dict(case_version_no=target.version_no, artifact_id=target.artifact_id,
                  historical=bool(historical), actor_kind="operator", actor_label=actor_label)
    rows = []
    if verdict is not None:
        rows.append(dict(event_type="operator_verdict", verdict=verdict, note=note, **common))
    for d in defects:
        rows.append(dict(event_type="defect", defect_code=d["code"], severity=d["severity"],
                         time_start_sec=d.get("start"), time_end_sec=d.get("end"),
                         note=d.get("note") or (note if verdict is None else None), **common))
    return _insert_events(session, key, request_sha, "feedback", case.id, rows)


def record_auto_assessment(session: Session, *, case_id: str, idempotency_key: str, case_version_no: int,
                           check_kind: str, claim: str, actor_kind: str = "system_check",
                           coverage: Optional[dict] = None, limitations: Optional[str] = None,
                           defect_code: Optional[str] = None, time_start_sec: Optional[float] = None,
                           time_end_sec: Optional[float] = None, note: Optional[str] = None,
                           import_batch_id: Optional[str] = None) -> Receipt:
    """Machine claim with explicit coverage; it is evidence, never a verdict (§5)."""
    _require_pg(session)
    key = _check_key(idempotency_key)
    _one_of(actor_kind, ("system_check", "assistant", "importer"), "automatic actor")
    try:
        t.validate_auto_assessment(check_kind, claim, defect_code)
    except ValueError as exc:
        raise KnowledgeInvalid(str(exc)) from None
    if defect_code is not None:
        _one_of(defect_code, t.DEFECT_CODES, "defect code")
    body = {"case_id": case_id, "v": case_version_no, "check": check_kind, "claim": claim, "actor": actor_kind,
            "coverage": coverage, "limitations": limitations, "defect": defect_code,
            "range": [time_start_sec, time_end_sec], "note": note, "batch": import_batch_id}
    request_sha = canonical_sha256(["auto_assessment", body])
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return replay
    version = session.execute(select(KbCaseVersion).where(KbCaseVersion.case_id == case_id,
                                                          KbCaseVersion.version_no == case_version_no)
                              ).scalar_one_or_none()
    if version is None:
        raise KnowledgeInvalid("unknown case version")
    case = session.get(KbCase, case_id)
    # Currentness is decided under the same Run → case locks every version
    # writer takes, so no append or withdrawal can slip in before the insert.
    lock_run_and_cases(session, case.run_id, [case.id])
    invalid = InvalidSet.load(session)
    if invalid.case_invalid(case) or invalid.version_invalid(version):
        raise KnowledgeInvalid("this case version was withdrawn")
    artifact = session.get(KbArtifact, version.artifact_id)
    current = latest_version(session, case_id, invalid)
    is_current = current is not None and current.id == version.id
    if not is_current and not bytes_preserved(session, artifact):
        # S10: a superseded mutable path may hold other bytes now.
        raise KnowledgeInvalid("this version's bytes were not preserved; it cannot be assessed after the fact")
    _validate_range(time_start_sec, time_end_sec, artifact.duration_sec)
    return _insert_events(session, key, request_sha, "auto_assessment", case_id, [dict(
        case_version_no=case_version_no, artifact_id=version.artifact_id, event_type="auto_assessment",
        historical=not is_current,
        actor_kind=actor_kind, check_kind=check_kind, claim=claim, coverage=coverage, limitations=limitations,
        defect_code=defect_code, time_start_sec=time_start_sec, time_end_sec=time_end_sec, note=note,
        import_batch_id=import_batch_id)])


def record_correction(session: Session, scope: KnowledgeScope, *, case_id: str, idempotency_key: str,
                      before_artifact_id: str, method: str, resolved: str,
                      after_artifact_id: Optional[str] = None, incremental_mcredits: Optional[int] = None,
                      cost_source: Optional[str] = None, operator_minutes: Optional[float] = None,
                      note: Optional[str] = None, actor_kind: str = "operator",
                      actor_label: Optional[str] = None, import_batch_id: Optional[str] = None) -> Receipt:
    _require_pg(session)
    key = _check_key(idempotency_key)
    _one_of(method, t.CORRECTION_METHODS, "correction method")
    _one_of(resolved, t.RESOLVED, "resolved")
    _one_of(actor_kind, t.ACTOR_KINDS, "actor")
    if incremental_mcredits is not None:
        if not isinstance(incremental_mcredits, int) or incremental_mcredits < 0:
            raise KnowledgeInvalid("incremental_mcredits must be a non-negative integer")
        if cost_source is None:
            raise KnowledgeInvalid("a correction cost needs its source")
    if cost_source is not None:
        _one_of(cost_source, t.COST_SOURCES, "cost source")
    _finite_nonneg(operator_minutes, "operator_minutes")
    invalid = InvalidSet.load(session)
    case = _scoped_case(session, scope, case_id, invalid)
    body = {"case_id": case_id, "before": before_artifact_id, "after": after_artifact_id, "method": method,
            "resolved": resolved, "mcredits": incremental_mcredits, "cost_source": cost_source,
            "minutes": operator_minutes, "note": note, "actor": actor_kind, "label": actor_label}
    request_sha = canonical_sha256(["correction", body])
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return replay
    # Both sides resolve server-side to this case: the before side is one of its
    # versions; the after side is one of its versions or a version of an external
    # repair linked to it. Anything else is unknown here (404, like out of scope).
    case_artifacts = {v.artifact_id for v in versions(session, case.id, invalid)}
    if before_artifact_id not in case_artifacts:
        raise KnowledgeAccessDenied("knowledge resource unavailable")
    if after_artifact_id is not None:
        linked = set(case_artifacts)
        for child in session.execute(select(KbCase).where(KbCase.parent_case_id == case.id,
                                                          KbCase.scope == "external")).scalars():
            if not invalid.case_invalid(child):
                linked |= {v.artifact_id for v in versions(session, child.id, invalid)}
        if after_artifact_id not in linked:
            raise KnowledgeAccessDenied("knowledge resource unavailable")
    return _insert_events(session, key, request_sha, "correction", case.id, [dict(
        event_type="correction", actor_kind=actor_kind, actor_label=actor_label, method=method,
        before_artifact_id=before_artifact_id, after_artifact_id=after_artifact_id, resolved=resolved,
        incremental_mcredits=incremental_mcredits, cost_source=cost_source,
        operator_minutes=operator_minutes, note=note, import_batch_id=import_batch_id)])


def retract_event(session: Session, scope: KnowledgeScope, *, event_id: str, idempotency_key: str,
                  reason: str, actor_kind: str = "operator", actor_label: Optional[str] = None) -> Receipt:
    """Exactly one retraction per target, same case; retractions are final (§4.5)."""
    _require_pg(session)
    key = _check_key(idempotency_key)
    if not reason or not reason.strip():
        raise KnowledgeInvalid("a retraction needs a reason")
    invalid = InvalidSet.load(session)
    target = session.get(KbFeedbackEvent, event_id)
    if target is None:
        raise KnowledgeAccessDenied("knowledge resource unavailable")
    case = _scoped_case(session, scope, target.case_id, invalid)
    request_sha = canonical_sha256(["retraction", {"event": event_id, "reason": reason, "actor": actor_kind,
                                                    "label": actor_label}])
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return replay
    if target.event_type == "retraction":
        raise KnowledgeInvalid("a retraction cannot be retracted; submit the judgement again instead")
    automatic = target.actor_kind != "operator"
    if automatic and actor_kind not in ("system_check", "importer"):
        raise KnowledgeInvalid("only the system or importer may retract an automatic event")
    if not automatic and actor_kind != "operator":
        raise KnowledgeInvalid("only an operator may retract an operator event")
    lock_run_and_cases(session, case.run_id if case.scope == "run" else None, [case.id])
    if session.execute(select(KbFeedbackEvent.id).where(KbFeedbackEvent.supersedes_event_id == event_id,
                                                        KbFeedbackEvent.event_type == "retraction")).first():
        raise KnowledgeConflict("this event is already retracted")
    if target.event_type == "operator_verdict" and not target.historical:
        # Retracting a verdict re-exposes the one before it; never let that be an
        # acceptance that contradicts a live blocker/major defect.
        remaining = [e for e in effective_events(session, case.id, invalid) if e.id != event_id]
        if _verdict_for(remaining, target.case_version_no) == "accepted_as_is" and any(
                e.event_type == "defect" and not e.historical and e.case_version_no == target.case_version_no
                and e.severity in ("blocker", "major") for e in remaining):
            raise KnowledgeConflict("retracting this would revive an acceptance over a live blocker/major defect; "
                                    "retract the defect or the acceptance first", target.case_version_no)
    return _insert_events(session, key, request_sha, "retraction", case.id, [dict(
        event_type="retraction", actor_kind=actor_kind, actor_label=actor_label, case_version_no=None,
        supersedes_event_id=event_id, historical=target.historical, note=reason)])


def record_export_observed(session: Session, *, case_id: str, idempotency_key: str, exported_at: datetime,
                           source_record: str, actor_kind: str = "importer",
                           import_batch_id: Optional[str] = None) -> Receipt:
    """A Final Creatives stamp as its own fact (§4.5): never an acceptance.

    Which version was exported is not recorded by the stamp, so the event names
    none; the stamp time and its source row go into the note.
    """
    _require_pg(session)
    key = _check_key(idempotency_key)
    _one_of(actor_kind, ("system_check", "importer"), "export observer")
    case = session.get(KbCase, case_id)
    if case is None:
        raise KnowledgeInvalid("unknown case")
    note = f"final_creatives_exported_at={exported_at.isoformat()} source={source_record}"
    request_sha = canonical_sha256(["export_observed", {"case_id": case_id, "note": note, "actor": actor_kind}])
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return replay
    # §6.2 hierarchy: key lock → Run row → case row. A derived/manual case is
    # Run-bound too (an export of a tail edit races a mutation of its Run).
    lock_run_and_cases(session, case.run_id, [case.id])
    return _insert_events(session, key, request_sha, "export_observed", case_id, [dict(
        event_type="export_observed", actor_kind=actor_kind, note=note, import_batch_id=import_batch_id)])


EXTERNAL_REPAIR_LOCATORS = ("drive_file", "local_path")


def record_external_repair(session: Session, scope: KnowledgeScope, *, run_id: str, idempotency_key: str,
                           locator_kind: str, locator: str, sha256: Optional[str] = None,
                           bytes: Optional[int] = None, duration_sec: Optional[float] = None,
                           replaced_delivered: Optional[bool] = None, note: Optional[str] = None,
                           actor_label: Optional[str] = None) -> Receipt:
    """S5: an operator-supplied repair made outside the pipelines (§7 item 5).

    receipt → Run lock → ``ensure_run_case`` → gap-free ``n`` under the lock →
    external case (version 1 = the repair) → ``link_external`` event on the Run
    case. Nothing is inferred: ``note`` stays the operator's text or NULL, and
    ``replaced_delivered`` is a declaration kept on the external case — whether
    the Run's file really changed is decided only by the done-gated sync hash.
    """
    from ..models import Run

    _require_pg(session)
    key = _check_key(idempotency_key)
    _one_of(locator_kind, EXTERNAL_REPAIR_LOCATORS, "external repair locator")
    if not isinstance(locator, str) or not locator.strip():
        raise KnowledgeInvalid("an external repair needs a Drive file or an uploaded file")
    if locator_kind == "drive_file" and sha256 is not None:
        raise KnowledgeInvalid("a Drive link is recorded unhashed")
    if locator_kind == "local_path" and sha256 is None:
        raise KnowledgeInvalid("an uploaded repair is recorded with its sha256")
    if replaced_delivered is not None and not isinstance(replaced_delivered, bool):
        raise KnowledgeInvalid("replaced_delivered must be true, false or absent")
    note = (note.strip() or None) if isinstance(note, str) else None
    invalid = InvalidSet.load(session)
    run = session.get(Run, run_id)
    root = current_case(session, run_case_key(run_id), invalid)
    project_id = root.project_id if root is not None else (run.project_id if run is not None else None)
    if run is None or project_id is None or not scope.allows_project(project_id):
        raise KnowledgeAccessDenied("knowledge resource unavailable")
    body = {"run_id": run_id, "locator_kind": locator_kind, "locator": locator, "sha256": sha256,
            "bytes": bytes, "duration_sec": duration_sec, "replaced_delivered": replaced_delivered,
            "note": note, "actor_label": actor_label}
    request_sha = canonical_sha256(["external_repair", body])
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return replay
    lock_run_and_cases(session, run_id, [])
    root = ensure_run_case(session, run_id)
    lock_run_and_cases(session, None, [root.id])
    # Gap-free under the Run lock: every revision of every earlier external
    # case of this Run keeps its number, withdrawn or not.
    prefix = external_case_key(run_id, 0)[:-1]
    n = (session.execute(select(func.count(func.distinct(KbCase.case_key)))
                         .where(KbCase.case_key.startswith(prefix, autoescape=True))).scalar() or 0) + 1
    artifact = get_or_create_artifact(session, kind="external_repair", locator_kind=locator_kind,
                                      locator=locator.strip(), sha256=sha256, project_id=root.project_id,
                                      bytes=bytes, duration_sec=duration_sec)
    if session.execute(select(KbCase.id).join(KbCaseVersion, KbCaseVersion.case_id == KbCase.id).where(
            KbCaseVersion.artifact_id == artifact.id, KbCase.parent_case_id == root.id,
            KbCase.scope == "external", ~withdrawn_clause("case", KbCase.id))).first():
        # One file is one repair: a second link would make the receipt ambiguous.
        raise KnowledgeConflict("this file is already linked as an external repair of this run")
    snapshot = {"linked_from_run_id": run_id}
    if replaced_delivered is not None:
        snapshot["replaced_delivered"] = replaced_delivered
    external = ensure_case(session, case_key=external_case_key(run_id, n), scope="external", operation="repair",
                           project_id=root.project_id, run_id=run_id, parent_case_id=root.id,
                           language=root.language, snapshot=snapshot)
    lock_run_and_cases(session, None, [external.id])
    # S4 adds the "external_repair" reason with its CHECK migration; until then
    # the repair is the external case's first output.
    reason = "external_repair" if "external_repair" in t.VERSION_REASONS else "initial_output"
    append_case_version(session, external, artifact_id=artifact.id, reason=reason, invalid=invalid)
    before = latest_version(session, root.id, invalid)
    return _insert_events(session, key, request_sha, "external_repair", root.id, [dict(
        event_type="link_external", actor_kind="operator", actor_label=actor_label,
        case_version_no=before.version_no if before is not None else None,
        before_artifact_id=before.artifact_id if before is not None else None,
        after_artifact_id=artifact.id, note=note)])


def external_case_for_event(session: Session, event_id: str) -> Optional[KbCase]:
    """The external case a ``link_external`` event created (receipts name the event only)."""
    event = session.get(KbFeedbackEvent, event_id)
    if event is None or event.event_type != "link_external" or event.after_artifact_id is None:
        return None
    return session.execute(select(KbCase).join(KbCaseVersion, KbCaseVersion.case_id == KbCase.id)
                           .where(KbCaseVersion.artifact_id == event.after_artifact_id,
                                  KbCase.parent_case_id == event.case_id, KbCase.scope == "external")
                           .order_by(KbCase.seq.desc()).limit(1)).scalar_one_or_none()


def _insert_events(session: Session, key: str, request_sha: str, kind: str, case_id: str,
                   rows: list[dict]) -> Receipt:
    ids = [str(uuid.uuid4()) for _ in rows]
    session.add(KbRequest(idempotency_key=key, request_sha256=request_sha, kind=kind, case_id=case_id,
                          child_event_ids=ids))
    session.flush()
    session.add_all([KbFeedbackEvent(id=i, request_id=key, ordinal=n, case_id=case_id, **row)
                     for n, (i, row) in enumerate(zip(ids, rows))])
    session.flush()
    return Receipt(key, kind, case_id, tuple(ids), False)


# ---------------------------------------------------------------- rules


def _latest_manifest(session: Session) -> Optional[KbEvalManifest]:
    """Highest manifest version; amendments only ever move holdout → development (§8.1)."""
    rows = session.execute(select(KbEvalManifest)).scalars().all()
    return max(rows, key=lambda m: (int(m.version) if m.version.isdigit() else -1, m.created_at),
               default=None)


def _holdout_project_ids(session: Session) -> set:
    manifest = _latest_manifest(session)
    if manifest is None:
        return set()
    return {p for g in manifest.body.get("groups", ()) if g.get("split") == "holdout"
            for p in g.get("project_ids", ())}


def is_holdout(session: Session, case: KbCase, holdout_projects: Optional[set] = None) -> bool:
    """Holdout under the latest manifest, by explicit membership or by project group (§8.2).

    Memberships under superseded manifest versions are history, not current split.
    """
    manifest = _latest_manifest(session)
    if manifest is None:
        return False
    if session.execute(select(KbEvalMembership.id).where(KbEvalMembership.case_id == case.id,
                                                         KbEvalMembership.manifest_version == manifest.version,
                                                         KbEvalMembership.split == "holdout")).first():
        return True
    projects = holdout_projects if holdout_projects is not None else _holdout_project_ids(session)
    return case.project_id in projects


def _validate_recommendation(rec) -> dict:
    if not isinstance(rec, dict) or set(rec) - {"action", "target"}:
        raise KnowledgeInvalid("recommendation must be {action, target}")
    _one_of(rec.get("action"), t.RULE_ACTIONS, "recommendation action")
    target = rec.get("target")
    if not target or not isinstance(target, (str, dict)):
        raise KnowledgeInvalid("recommendation target is required")
    return {"action": rec["action"], "target": target}


SHAREABLE_TARGET_KEYS = frozenset(t.APPLICABILITY_DIMENSIONS) - {"project_id", "subject_id"}


def _validate_shareable_structure(applicability: dict, recommendation: dict) -> None:
    """Shareable text is crossing scope, so its structured parts may not name a
    project or a person: no project/subject applicability, and a structured
    target over generic dimensions only (no free-text target)."""
    if {"project_id", "subject_id"} & set(applicability):
        raise KnowledgeInvalid("a shareable rule cannot be scoped to a project or subject")
    target = recommendation["target"]
    if not isinstance(target, dict) or not target or set(target) - SHAREABLE_TARGET_KEYS:
        raise KnowledgeInvalid("a shareable rule needs a structured target over generic dimensions")
    for value in target.values():
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(v, (str, bool)) or v == "" for v in values):
            raise KnowledgeInvalid("shareable target values must be plain codes")


def add_rule_version(session: Session, scope: KnowledgeScope, *, slug: str, idempotency_key: str, kind: str,
                     statement: str, recommendation: dict, confidence: str, author_kind: str,
                     effective_from: datetime, rationale: Optional[str] = None,
                     applicability: Optional[dict] = None, shareable: bool = False,
                     scope_project_ids: Optional[list] = None, model_registry_revision: Optional[str] = None,
                     effective_to: Optional[datetime] = None, author_label: Optional[str] = None,
                     provenance: Optional[dict] = None, evidence_gaps: Optional[list] = None,
                     evidence: Iterable[dict] = (), import_batch_id: Optional[str] = None,
                     require_new: bool = False) -> KbRuleVersion:
    """New version of ``slug`` (rule created on first use) with cited evidence.

    Versions are numbered under the rule row lock; the previous highest version
    is recorded as superseded. Evidence must be visible and never holdout.
    """
    _require_pg(session)
    key = _check_key(idempotency_key)
    _one_of(kind, t.RULE_KINDS, "rule kind")
    _one_of(confidence, t.RULE_CONFIDENCE, "confidence")
    _one_of(author_kind, t.AUTHOR_KINDS, "author kind")
    if kind == "operator_constraint" and author_kind not in ("operator", "importer"):
        raise KnowledgeInvalid("only an operator can state an operator constraint")
    if not statement or not statement.strip():
        raise KnowledgeInvalid("statement is required")
    try:
        applicability = t.canonical_applicability(applicability)
    except ValueError as exc:
        raise KnowledgeInvalid(str(exc)) from None
    recommendation = _validate_recommendation(recommendation)
    if shareable:
        _validate_shareable_structure(applicability, recommendation)
    if effective_from.tzinfo is None or (effective_to is not None and effective_to.tzinfo is None):
        raise KnowledgeInvalid("effective window must be timezone-aware")
    if effective_to is not None and effective_to <= effective_from:
        raise KnowledgeInvalid("effective_to must be after effective_from")
    if scope_project_ids is not None:
        if not scope_project_ids:
            raise KnowledgeInvalid("scope_project_ids must be NULL (global) or non-empty")
        scope_project_ids = sorted(set(scope_project_ids))
        for pid in scope_project_ids:
            scope.require_project(pid)
    evidence = [dict(e) for e in evidence]
    body = {k: v for k, v in locals().items() if k in (
        "slug", "kind", "statement", "rationale", "applicability", "recommendation", "confidence", "shareable",
        "scope_project_ids", "model_registry_revision", "effective_from", "effective_to", "author_kind",
        "author_label", "provenance", "evidence_gaps", "evidence", "import_batch_id", "require_new")}
    request_sha = canonical_sha256(["rule_version", body])
    # Scope and evidence first: a replay must not reveal another project's receipt.
    invalid = InvalidSet.load(session)
    if evidence and _latest_manifest(session) is None:
        raise KnowledgeInvalid("load the evaluation manifest before citing evidence (holdout cannot be checked)")
    holdout = _holdout_project_ids(session)
    checked = []
    for e in evidence:
        _one_of(e.get("stance"), t.EVIDENCE_STANCES, "evidence stance")
        case = _scoped_case(session, scope, e.get("case_id"), invalid)
        if is_holdout(session, case, holdout):
            raise KnowledgeInvalid("holdout cases cannot be cited as rule evidence")
        if e.get("event_id") is not None:
            ev = session.get(KbFeedbackEvent, e["event_id"])
            # Only an effective event can support a rule: not retracted, not withdrawn
            # directly, through its artifact, or through its case version.
            if ev is None or ev.case_id != case.id or ev.id not in {
                    x.id for x in effective_events(session, case.id, invalid)}:
                raise KnowledgeInvalid("cited event must be an effective event of the cited case")
        checked.append(e)
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return session.get(KbRuleVersion, replay.ids[0])
    session.execute(pg_insert(KbRule).values(id=str(uuid.uuid4()), slug=slug)
                    .on_conflict_do_nothing(constraint="uq_kb_rule_slug"))
    rule = session.execute(select(KbRule).where(KbRule.slug == slug).with_for_update()).scalar_one()
    previous = session.execute(select(KbRuleVersion).where(KbRuleVersion.rule_id == rule.id)
                               .order_by(KbRuleVersion.version_no.desc()).limit(1)).scalar_one_or_none()
    if require_new and previous is not None:
        raise KnowledgeConflict("rule slug already exists; add a version instead")
    version = KbRuleVersion(
        rule_id=rule.id, version_no=(previous.version_no + 1) if previous else 1, kind=kind,
        statement=statement, rationale=rationale, applicability=applicability, recommendation=recommendation,
        confidence=confidence, shareable=bool(shareable), scope_project_ids=scope_project_ids,
        model_registry_revision=model_registry_revision, effective_from=effective_from,
        effective_to=effective_to, author_kind=author_kind, author_label=author_label,
        provenance=provenance or {}, evidence_gaps=evidence_gaps or [],
        supersedes_version_id=previous.id if previous else None, import_batch_id=import_batch_id)
    session.add(version)
    session.flush()
    session.add_all([KbRuleEvidence(rule_version_id=version.id, case_id=e["case_id"], event_id=e.get("event_id"),
                                    stance=e["stance"], note=e.get("note"), import_batch_id=import_batch_id)
                     for e in checked])
    _store_request(session, key, request_sha, "rule_version", None, [version.id])
    return version


def review_rule_version(session: Session, scope: KnowledgeScope, *, rule_version_id: str, status: str,
                        idempotency_key: str, reviewer_label: Optional[str] = None,
                        note: Optional[str] = None) -> KbRuleReview:
    _require_pg(session)
    key = _check_key(idempotency_key)
    _one_of(status, t.REVIEW_STATUSES, "review status")
    request_sha = canonical_sha256(["rule_review", rule_version_id, status, reviewer_label, note])
    version = session.get(KbRuleVersion, rule_version_id)
    if version is None or rule_visibility(session, scope, version) != "full":
        raise KnowledgeAccessDenied("knowledge resource unavailable")
    replay = _replay(session, key, request_sha)
    if replay is not None:
        return session.get(KbRuleReview, replay.ids[0])
    if status == "approved" and version.kind == "machine_hypothesis":
        raise KnowledgeInvalid("a machine hypothesis cannot be approved; restate it as guidance")
    session.execute(select(KbRule.id).where(KbRule.id == version.rule_id).with_for_update())
    review = KbRuleReview(rule_version_id=version.id, status=status, reviewer_label=reviewer_label, note=note)
    session.add(review)
    session.flush()
    _store_request(session, key, request_sha, "rule_review", None, [review.id])
    return review


def _evidence_projects(session: Session, version_id: str) -> list:
    return [p for (p,) in session.execute(select(KbCase.project_id).join(
        KbRuleEvidence, KbRuleEvidence.case_id == KbCase.id).where(KbRuleEvidence.rule_version_id == version_id)
    ).all()]


def rule_visibility(session: Session, scope: KnowledgeScope, version: KbRuleVersion) -> Optional[str]:
    return scope.rule_visibility(shareable=version.shareable, scope_project_ids=version.scope_project_ids,
                                 evidence_project_ids=_evidence_projects(session, version.id))


def effective_rule_versions(session: Session, at: Optional[datetime] = None,
                            invalid: Optional[InvalidSet] = None) -> list[KbRuleVersion]:
    """Per rule, the highest version ever approved, if its latest review is still
    ``approved`` and its window contains ``at`` (§4.6). Retiring it never
    resurrects an older version."""
    at = at or datetime.now(timezone.utc)
    invalid = invalid if invalid is not None else InvalidSet.load(session)
    reviews = session.execute(select(KbRuleReview).order_by(KbRuleReview.seq)).scalars().all()
    latest_review, ever_approved = {}, set()
    for r in reviews:
        latest_review[r.rule_version_id] = r.status
        if r.status == "approved":
            ever_approved.add(r.rule_version_id)
    candidates: dict = {}
    for v in session.execute(select(KbRuleVersion).where(KbRuleVersion.id.in_(ever_approved))).scalars():
        if v.kind == "machine_hypothesis" or invalid.row_invalid("rule_version", v):
            continue
        if v.rule_id not in candidates or v.version_no > candidates[v.rule_id].version_no:
            candidates[v.rule_id] = v
    holdout = _holdout_project_ids(session)
    return sorted((v for v in candidates.values()
                   if latest_review.get(v.id) == "approved" and v.effective_from <= at
                   and (v.effective_to is None or at < v.effective_to)
                   and not _cites_holdout(session, v.id, holdout)), key=lambda v: v.seq)


def _cites_holdout(session: Session, version_id: str, holdout_projects: set) -> bool:
    """Defence in depth for evidence written before a manifest was loaded (§8.2)."""
    for case_id, project_id in session.execute(
            select(KbCase.id, KbCase.project_id).join(KbRuleEvidence, KbRuleEvidence.case_id == KbCase.id)
            .where(KbRuleEvidence.rule_version_id == version_id)).all():
        if project_id in holdout_projects or is_holdout(session, session.get(KbCase, case_id), holdout_projects):
            return True
    return False


def rule_evidence(session: Session, version_id: str, invalid: Optional[InvalidSet] = None) -> list:
    invalid = invalid if invalid is not None else InvalidSet.load(session)
    rows = session.execute(select(KbRuleEvidence).where(KbRuleEvidence.rule_version_id == version_id)
                           .order_by(KbRuleEvidence.seq)).scalars().all()
    out, effective = [], {}
    for e in rows:
        case = session.get(KbCase, e.case_id)
        if invalid.row_invalid("rule_evidence", e) or invalid.case_invalid(case):
            continue
        if e.event_id is not None:
            if case.id not in effective:
                effective[case.id] = {ev.id for ev in effective_events(session, case.id, invalid)}
            if e.event_id not in effective[case.id]:
                continue  # retracted or invalidated judgement no longer supports anything
        out.append(e)
    return out


# ---------------------------------------------------------------- decisions


def record_decision(session: Session, scope: KnowledgeScope, *, idempotency_key: str, mode: str,
                    requested_by: str, context: dict, advisor_version: str, recommendation: dict,
                    consulted_rule_version_ids: Iterable[str] = (), consulted_case_ids: Iterable[str] = (),
                    retrieval_params: Optional[dict] = None, project_id: Optional[str] = None) -> KbDecision:
    """Pin exactly what was consulted. Only the trusted advisor calls this; every
    id is re-checked against scope, never trusted from a model (§4.7, §6.5)."""
    _require_pg(session)
    key = _check_key(idempotency_key)
    _one_of(mode, t.DECISION_MODES, "decision mode")
    _one_of(requested_by, t.DECISION_REQUESTERS, "decision requester")
    if project_id is not None:
        scope.require_project(project_id)
    rule_ids = sorted(set(consulted_rule_version_ids))
    case_ids = sorted(set(consulted_case_ids))
    invalid = InvalidSet.load(session)
    holdout = _holdout_project_ids(session)
    pinned = []
    for rid in rule_ids:
        v = session.get(KbRuleVersion, rid)
        if v is None or rule_visibility(session, scope, v) is None:
            raise KnowledgeAccessDenied("knowledge resource unavailable")
        if invalid.row_invalid("rule_version", v):
            raise KnowledgeConflict("a consulted rule version has been withdrawn; retrieve again")
        if _cites_holdout(session, v.id, holdout):
            raise KnowledgeConflict("a consulted rule version cites holdout evidence and is never advisable")
        pinned.append([v.id, v.rule_id, v.version_no, canonical_sha256([v.statement, v.applicability,
                                                                         v.recommendation, v.kind])])
    for cid in case_ids:
        if is_holdout(session, _scoped_case(session, scope, cid, invalid), holdout):
            raise KnowledgeConflict("a consulted case is holdout and is never advisable")
    consulted_sha = canonical_sha256({"rules": pinned, "cases": case_ids, "context": context,
                                      "recommendation": recommendation, "advisor": advisor_version})
    _advisory_lock(session, "kb_decision", key)
    existing = session.execute(select(KbDecision).where(KbDecision.idempotency_key == key)).scalar_one_or_none()
    if existing is not None:
        if existing.consulted_sha256 != consulted_sha:
            raise KnowledgeConflict("decision idempotency key already used for a different decision")
        return existing
    decision = KbDecision(idempotency_key=key, mode=mode, requested_by=requested_by, project_id=project_id,
                          context=context, advisor_version=advisor_version, retrieval_params=retrieval_params or {},
                          consulted_rule_version_ids=rule_ids, consulted_case_ids=case_ids,
                          consulted_sha256=consulted_sha, recommendation=recommendation)
    session.add(decision)
    session.flush()
    return decision


def record_decision_outcome(session: Session, scope: KnowledgeScope, *, decision_id: str, chosen: dict,
                            overrode: bool, override_note: Optional[str] = None,
                            attempt_id: Optional[str] = None) -> KbDecisionOutcome:
    decision = session.get(KbDecision, decision_id)
    if decision is None or (decision.project_id is not None and not scope.allows_project(decision.project_id)):
        raise KnowledgeAccessDenied("knowledge resource unavailable")
    if attempt_id is not None:
        attempt = session.get(KbAttempt, attempt_id)
        case = session.get(KbCase, attempt.case_id) if attempt else None
        if case is None or not scope.allows_project(case.project_id):
            raise KnowledgeAccessDenied("knowledge resource unavailable")
        if decision.project_id is not None and case.project_id != decision.project_id:
            raise KnowledgeInvalid("the outcome attempt must belong to the decision's project")
    outcome = KbDecisionOutcome(decision_id=decision_id, chosen=chosen, overrode=bool(overrode),
                                override_note=override_note, attempt_id=attempt_id)
    session.add(outcome)
    session.flush()
    return outcome


# ---------------------------------------------------------------- evaluation manifest


def manifest_sha256(body: dict) -> str:
    """Same canonical form as scripts/knowledge_holdout_split.py."""
    content = {k: v for k, v in body.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_eval_manifest(session: Session, raw: bytes, *, expected_sha256: str) -> KbEvalManifest:
    _require_pg(session)
    body = json.loads(raw)
    sha = manifest_sha256(body)
    if sha != expected_sha256 or body.get("manifest_sha256") != sha:
        raise KnowledgeInvalid("manifest content does not match its pinned sha256")
    version = str(body["version"])
    session.execute(pg_insert(KbEvalManifest).values(version=version, sha256=sha, body=body)
                    .on_conflict_do_nothing(index_elements=["version"]))
    stored = session.get(KbEvalManifest, version)
    if stored.sha256 != sha:
        raise KnowledgeConflict(f"manifest version {version} is already loaded with different content")
    return stored


def assign_eval_membership(session: Session, case_id: str, manifest_version: str,
                           import_batch_id: Optional[str] = None) -> KbEvalMembership:
    """Split and group come from the stored manifest by the case's project, never from input."""
    manifest = session.get(KbEvalManifest, manifest_version)
    case = session.get(KbCase, case_id)
    if manifest is None or case is None:
        raise KnowledgeInvalid("unknown manifest or case")
    group = next((g for g in manifest.body.get("groups", ()) if case.project_id in g.get("project_ids", ())), None)
    if group is None:
        raise KnowledgeInvalid("case project is not in the manifest (excluded from evaluation)")
    session.execute(pg_insert(KbEvalMembership).values(
        id=str(uuid.uuid4()), case_id=case_id, manifest_version=manifest_version, split=group["split"],
        group_key=group["group_key"], import_batch_id=import_batch_id)
        .on_conflict_do_nothing(constraint="uq_kb_eval_membership"))
    return session.execute(select(KbEvalMembership).where(KbEvalMembership.case_id == case_id,
                                                          KbEvalMembership.manifest_version == manifest_version)
                           ).scalar_one()


# ---------------------------------------------------------------- subjects


def link_subject(session: Session, scope: KnowledgeScope, *, label: str, project_id: str,
                 segment_def_id: Optional[str] = None, time_start_sec: Optional[float] = None,
                 time_end_sec: Optional[float] = None, note: Optional[str] = None) -> KbSubjectLink:
    """Name the person in a source range; overlapping links to another subject are refused (§9.3)."""
    _require_pg(session)
    scope.require_project(project_id)
    if not label or not label.strip():
        raise KnowledgeInvalid("subject label is required")
    _validate_range(time_start_sec, time_end_sec, None)
    session.execute(pg_insert(KbSourceSubject).values(id=str(uuid.uuid4()), label=label.strip(), note=note)
                    .on_conflict_do_nothing(constraint="uq_kb_subject_label"))
    subject = session.execute(select(KbSourceSubject).where(KbSourceSubject.label == label.strip())).scalar_one()
    _advisory_lock(session, "kb_subject_project", project_id)
    lo, hi = (time_start_sec, time_end_sec) if time_start_sec is not None else (0.0, math.inf)
    for link in session.execute(select(KbSubjectLink).where(KbSubjectLink.project_id == project_id)).scalars():
        if link.segment_def_id is not None and segment_def_id is not None and link.segment_def_id != segment_def_id:
            continue
        a, b = (link.time_start_sec, link.time_end_sec) if link.time_start_sec is not None else (0.0, math.inf)
        if a < hi and lo < b:
            if link.subject_id != subject.id:
                raise KnowledgeConflict("this range is already linked to a different subject")
            if (a, b, link.segment_def_id) == (lo, hi, segment_def_id) or (
                    link.time_start_sec is None and time_start_sec is None
                    and link.segment_def_id == segment_def_id):
                return link
    link = KbSubjectLink(subject_id=subject.id, project_id=project_id, segment_def_id=segment_def_id,
                         time_start_sec=time_start_sec, time_end_sec=time_end_sec)
    session.add(link)
    session.flush()
    return link


# ---------------------------------------------------------------- deliverable capture (C1 rows / C2)


def capture_key(run_id: str, artifact_id: str, artifact_sha256: str, publication_id: str,
                attempt_no: int = 0) -> str:
    # The artifact id pins the revision: a replacement for a withdrawn artifact is
    # new work; attempt_no > 0 re-queues a publication whose result was withdrawn.
    parts = [run_id, artifact_id, artifact_sha256, publication_id] + ([attempt_no] if attempt_no else [])
    return canonical_sha256(parts)


def publication_covered(session: Session, run_id: str, publication_id: str) -> bool:
    """Is this publication already captured for the Run's current case?

    The scanner's "pending" test: True when a valid version of the current case
    carries the publication (or its re-record), or a queue item for it still
    covers it. A withdrawn result makes it pending again, so the next
    ``enqueue_final_capture`` re-queues it for the replacement case revision.
    """
    invalid = InvalidSet.load(session)
    case = current_case(session, run_case_key(run_id), invalid)
    if case is not None and valid_publication_version(session, case.id, publication_id, invalid) is not None:
        return True
    items = session.execute(select(KbCaptureQueue).where(KbCaptureQueue.run_id == run_id,
                                                         KbCaptureQueue.source_record == publication_id)
                            ).scalars().all()
    return any(not invalid.has("artifact", i.artifact_id) and _capture_still_covers(session, i) for i in items)


def _capture_still_covers(session: Session, item: KbCaptureQueue) -> bool:
    """True unless the item was consumed and what it produced is no longer valid
    for the Run's current case (case or version withdrawn)."""
    done = session.execute(select(KbCaptureDone).where(KbCaptureDone.queue_id == item.id)).scalar_one_or_none()
    if done is None or done.outcome in ("stale", "withdrawn"):
        return True  # pending, or deliberately not current: nothing to redo
    invalid = InvalidSet.load(session)
    if invalid.has("artifact", item.artifact_id):
        return False
    case = current_case(session, run_case_key(item.run_id), invalid)
    if case is None:
        return False
    if valid_publication_version(session, case.id, item.source_record, invalid) is not None:
        return True
    if done.outcome == "duplicate":
        latest = latest_version(session, case.id, invalid)
        return latest is not None and session.get(KbArtifact, latest.artifact_id).sha256 == item.artifact_sha256
    return False


def file_identity(path: str) -> Optional[str]:
    """"dev:ino:mtime_ns" of a file, or None if it cannot be stat-ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{st.st_dev}:{st.st_ino}:{st.st_mtime_ns}"


def enqueue_final_capture(session: Session, *, run_id: str, artifact_id: str, reason: str,
                          publication_id: str, file_identity: str) -> KbCaptureQueue:
    """C1's database half: an idempotent queue item; no Run lock (§6.1 C1).

    ``file_identity`` ("dev:ino:mtime_ns" of the published file) is required:
    C2 lets an item become current only if it still names the live final.
    """
    _require_pg(session)
    if not isinstance(file_identity, str) or not file_identity:
        raise KnowledgeInvalid("a captured final needs the identity of the published file")
    _check_record(publication_id)
    _one_of(reason, t.VERSION_REASONS, "version reason")
    from ..models import Run

    artifact = session.get(KbArtifact, artifact_id)
    if artifact is None or artifact.sha256 is None:
        raise KnowledgeInvalid("a captured final must be a hashed artifact")
    if InvalidSet.load(session).has("artifact", artifact.id):
        raise KnowledgeInvalid("this artifact was withdrawn; record the final again")
    run = session.get(Run, run_id)
    case = current_case(session, run_case_key(run_id))
    owner = run.project_id if run is not None else (case.project_id if case is not None else None)
    if owner is None or artifact.project_id != owner:
        raise KnowledgeInvalid("a captured final must belong to the Run's project")
    # A consumed item whose result was later withdrawn (its case or version
    # invalidated) no longer covers this publication: queue it again under the
    # next key, so the replacement case revision receives it.
    _advisory_lock(session, "kb_capture", run_id, publication_id)
    for attempt_no in itertools.count():
        ckey = capture_key(run_id, artifact.id, artifact.sha256, publication_id, attempt_no)
        existing = session.execute(select(KbCaptureQueue).where(KbCaptureQueue.capture_key == ckey)
                                   ).scalar_one_or_none()
        if existing is None:
            session.execute(pg_insert(KbCaptureQueue).values(
                id=str(uuid.uuid4()), capture_key=ckey, run_id=run_id, artifact_id=artifact_id,
                artifact_sha256=artifact.sha256, reason=reason, source_record=publication_id,
                file_identity=file_identity).on_conflict_do_nothing(constraint="uq_kb_capture_key"))
            return session.execute(select(KbCaptureQueue).where(KbCaptureQueue.capture_key == ckey)).scalar_one()
        if _capture_still_covers(session, existing):
            return existing
    raise AssertionError("unreachable")


def ensure_run_case(session: Session, run_id: str) -> Optional[KbCase]:
    """Run case from the live Run row; caller already holds the Run lock."""
    from ..models import Run, VideoProject as Project

    run = session.execute(select(Run).where(Run.id == run_id)
                          .execution_options(populate_existing=True)).scalar_one_or_none()
    if run is None:
        return current_case(session, run_case_key(run_id))
    project = session.get(Project, run.project_id)
    operation = getattr(project, "project_type", None) or "face_swap"
    if operation not in t.CASE_OPERATIONS:
        operation = "face_swap"
    return ensure_case(session, case_key=run_case_key(run_id), scope="run", operation=operation,
                       project_id=run.project_id, run_id=run_id, language=getattr(run, "language", None),
                       snapshot={"run_name": getattr(run, "name", None)})


def _locked_result_path(session: Session, run_id: str) -> Optional[str]:
    """The Run's final path read fresh from the database (never the identity
    map), so a caller holding the Run lock sees what was committed before it."""
    from ..models import Run

    return session.execute(select(Run.result_local_path).where(Run.id == run_id)
                           .execution_options(populate_existing=True)).scalar()


def _live_identity(session: Session, run_id: str) -> Optional[str]:
    path = _locked_result_path(session, run_id)
    return file_identity(path) if path else None


def drain_run_captures(session: Session, run_id: str, *, max_items: Optional[int] = None) -> list[KbCaptureDone]:
    """C2: allocate versions per Run in ``c1_seq`` order under the Run lock (§6.1 C2).

    ``max_items`` bounds one call (the oldest items first), so a backlog never
    holds the Run row across all of it; the rest drains on the next call.
    """
    _require_pg(session)
    lock_run_and_cases(session, run_id, [])
    case = ensure_run_case(session, run_id)
    if case is None:
        return []
    lock_run_and_cases(session, None, [case.id])
    pending = session.execute(
        select(KbCaptureQueue).where(KbCaptureQueue.run_id == run_id,
                                     ~exists().where(KbCaptureDone.queue_id == KbCaptureQueue.id))
        .order_by(KbCaptureQueue.c1_seq).limit(max_items)).scalars().all()
    done = []
    invalid = InvalidSet.load(session)
    # Newest appended item for this Run: computed once (one join over this Run's
    # queue via ix_kb_capture_queue_run), then advanced locally as items append.
    newest_c1 = session.execute(
        select(func.max(KbCaptureQueue.c1_seq)).join(KbCaptureDone, KbCaptureDone.queue_id == KbCaptureQueue.id)
        .where(KbCaptureQueue.run_id == run_id, KbCaptureDone.outcome == "appended")).scalar()
    for item in pending:
        latest = latest_version(session, case.id, invalid)  # withdrawn versions are not "current"
        if invalid.has("artifact", item.artifact_id):
            row = KbCaptureDone(queue_id=item.id, outcome="withdrawn")  # consumed, never current
        elif (latest is not None and session.get(KbArtifact, latest.artifact_id).sha256 == item.artifact_sha256) \
                or valid_publication_version(session, case.id, item.source_record, invalid) is not None:
            row = KbCaptureDone(queue_id=item.id, outcome="duplicate")  # same bytes or same publication
        elif item.file_identity != _live_identity(session, run_id):
            # Published, but no longer the live final (a later promotion or
            # restitch won the Run lock first): history, never current.
            version = append_case_version(session, case, artifact_id=item.artifact_id, reason=item.reason,
                                          source_record=item.source_record, historical=True,
                                          published_at=item.created_at, invalid=invalid)
            row = KbCaptureDone(queue_id=item.id, outcome="historical", case_version_no=version.version_no)
        elif newest_c1 is not None and newest_c1 > item.c1_seq:
            row = KbCaptureDone(queue_id=item.id, outcome="stale")
        else:
            version = append_case_version(session, case, artifact_id=item.artifact_id,
                                          reason=item.reason if latest is not None or item.reason != "restitch"
                                          else "initial_output", source_record=item.source_record,
                                          invalid=invalid)
            row = KbCaptureDone(queue_id=item.id, outcome="appended", case_version_no=version.version_no)
            newest_c1 = max(newest_c1 or 0, item.c1_seq)
        session.add(row)
        session.flush()
        done.append(row)
    return done


# ---------------------------------------------------------------- read model


HOLDOUT_HIDDEN_EVENTS = ("operator_verdict", "defect", "correction", "link_external", "retraction")


def _case_view(session: Session, case: KbCase, invalid: InvalidSet, holdout: bool = False) -> dict:
    """Holdout cases keep their timeline but never show operator judgements (§8.2 blinding)."""
    all_events = session.execute(select(KbFeedbackEvent).where(KbFeedbackEvent.case_id == case.id)
                                 .order_by(KbFeedbackEvent.seq)).scalars().all()
    if holdout:
        all_events = [e for e in all_events if e.event_type not in HOLDOUT_HIDDEN_EVENTS]
    effective_ids = {e.id for e in effective_events(session, case.id, invalid)}
    attempts = [a for a in session.execute(select(KbAttempt).where(KbAttempt.case_id == case.id)
                                           .order_by(KbAttempt.seq)).scalars()
                if not invalid.attempt_invalid(a)]
    return {
        "id": case.id, "case_key": case.case_key, "scope": case.scope, "operation": case.operation,
        "parent_case_id": case.parent_case_id, "origin": case.origin, "language": case.language,
        "time_range": [case.time_start_sec, case.time_end_sec], "missing_facts": case.missing_facts,
        "holdout": holdout,
        "state": ({"latest_version_no": case_state(session, case.id, invalid)["latest_version_no"],
                   "verdict": "hidden_holdout", "accepted_artifact_id": None,
                   "previous_accepted_version_no": None}
                  if holdout else case_state(session, case.id, invalid)),
        "versions": [{"version_no": v.version_no, "artifact_id": v.artifact_id, "reason": v.reason,
                      "source_record": v.source_record, "observed_at": v.observed_at.isoformat(),
                      "historical": v.historical,
                      "published_at": v.published_at.isoformat() if v.published_at else None,
                      "sha256": session.get(KbArtifact, v.artifact_id).sha256,
                      "invalidated": invalid.version_invalid(v)} for v in versions(session, case.id)],
        "attempts": [{"id": a.id, "kind": a.attempt_kind, "origin": a.origin, "provider": a.provider,
                      "model_key": a.model_key, "provider_model_id": a.provider_model_id,
                      **{k: v for k, v in attempt_state(session, a.id, invalid).items()
                         if k != "observations"}}
                     for a in attempts],
        "events": [{"id": e.id, "seq": e.seq, "type": e.event_type, "actor_kind": e.actor_kind,
                    "actor_label": e.actor_label, "actor_verified": False, "version_no": e.case_version_no,
                    "historical": e.historical, "verdict": e.verdict, "defect_code": e.defect_code,
                    "severity": e.severity, "time_range": [e.time_start_sec, e.time_end_sec],
                    "check_kind": e.check_kind, "claim": e.claim, "coverage": e.coverage,
                    "limitations": e.limitations, "method": e.method, "resolved": e.resolved,
                    "note": e.note, "supersedes_event_id": e.supersedes_event_id,
                    "effective": e.id in effective_ids, "invalidated": invalid.event_invalid(e)}
                   for e in all_events],
    }


def run_timeline(session: Session, scope: KnowledgeScope, run_id: str) -> dict:
    """Run case and every child case, scoped by the stored project (§6.6)."""
    from ..models import Run

    _require_pg(session)
    invalid = InvalidSet.load(session)
    run = session.get(Run, run_id)
    root = current_case(session, run_case_key(run_id), invalid)
    # Withdrawn knowledge never authorizes anything: a deleted Run with no
    # current case is simply unknown.
    project_id = root.project_id if root is not None else (run.project_id if run is not None else None)
    if project_id is None or not scope.allows_project(project_id):
        raise KnowledgeAccessDenied("knowledge resource unavailable")
    if root is None:
        return {"run_id": run_id, "cases": []}
    # Children stay attached to whichever root revision they were created under.
    revisions = select(KbCase.id).where(KbCase.case_key == run_case_key(run_id))
    children = session.execute(select(KbCase).where(KbCase.parent_case_id.in_(revisions))
                               .order_by(KbCase.seq)).scalars().all()
    cases = [root] + [c for c in children if not invalid.case_invalid(c) and scope.allows_project(c.project_id)]
    holdout = _holdout_project_ids(session)
    return {"run_id": run_id, "cases": [_case_view(session, c, invalid, is_holdout(session, c, holdout))
                                        for c in cases]}
