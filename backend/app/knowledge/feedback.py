"""Operator feedback service for the Run-page "Outcome" card (AICP-299).

Design: docs/knowledge-feedback-design.md §4.5, §6.2, §6.6, §7 and
docs/knowledge-capture-hooks.md §8 (disk guard, unpreserved versions) and §10
(card, start bootstrap). The JSON API (``knowledge.api``) and the HTMX twins
(``knowledge.web``) both call these functions, so the two surfaces cannot
drift apart.

Every write is one transaction with ``lock_timeout = 5s`` on the store
functions (lock order: idempotency key → ``runs`` row → ``kb_cases``). File
hashing happens BEFORE that transaction: a memo hasher hands the store the
pre-computed sha when the file's stat identity is unchanged under the lock,
and only re-hashes when the final moved in between.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from . import store
from .access import KnowledgeAccessDenied, KnowledgeScope

LOCK_TIMEOUT = "5s"
# Postgres: lock_not_available, query_canceled (statement/lock timeout), deadlock_detected.
_BUSY_PGCODES = {"55P03", "57014", "40P01"}

UPLOAD_MAX_BYTES = 500 << 20          # capture hooks rev. 2 §9.4 (nginx caps this route at 501m)
MULTIPART_OVERHEAD_BYTES = 1 << 20    # form fields + boundaries around the file part
UPLOAD_EXTENSIONS = (".mp4", ".mov", ".m4v")
MIN_FREE_BYTES = 20 << 30              # capture hooks §8 disk guard: free − n ≥ max(20 GB, 10 %)
MIN_FREE_FRACTION = 0.10

# The capture agent (S14) supplies the live-link lookup; until then sync records
# the live file as an unpreserved ``local_path`` version.
live_link_lookup: Optional[Callable] = None


class FeedbackError(Exception):
    """A failed feedback request, already classified for both surfaces."""

    def __init__(self, status: int, message: str, *, kind: str, current_version_no: Optional[int] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind
        self.current_version_no = current_version_no


def _not_found() -> FeedbackError:
    return FeedbackError(404, "Knowledge resource unavailable", kind="not_found")


# ---------------------------------------------------------------- hashing


@dataclass(frozen=True)
class PreHash:
    path: str
    sha256: str
    identity: tuple


def _identity(path: str) -> Optional[tuple]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def prehash(path: Optional[str]) -> Optional[PreHash]:
    """Hash a file outside any transaction; None when it is not a readable file."""
    if not path or not os.path.isfile(path):
        return None
    before = _identity(path)
    try:
        sha = store.sha256_file(path)
    except OSError:
        return None
    if before is None or _identity(path) != before:
        return None  # rewritten while hashing: let the locked path hash it
    return PreHash(path, sha, before)


def memo_hasher(pre: Optional[PreHash]) -> Callable[[str], str]:
    """The cached sha when the stat identity still matches under the lock, else a re-hash."""
    def hasher(path: str) -> str:
        if pre is not None and path == pre.path and _identity(path) == pre.identity:
            return pre.sha256
        return store.sha256_file(path)
    return hasher


# ---------------------------------------------------------------- transactions


def _run(session: Session, fn, *args, **kwargs):
    """One transaction with the 5 s lock timeout; every failure classified."""
    try:
        session.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
        result = fn(session, *args, **kwargs)
        session.commit()
        return result
    except KnowledgeAccessDenied:
        session.rollback()
        raise _not_found() from None
    except store.NotDelivered as exc:
        session.rollback()
        raise FeedbackError(409, str(exc), kind="not_delivered", current_version_no=exc.current_version_no) from None
    except store.KnowledgeConflict as exc:
        session.rollback()
        raise FeedbackError(409, str(exc), kind="conflict", current_version_no=exc.current_version_no) from None
    except store.KnowledgeInvalid as exc:
        session.rollback()
        raise FeedbackError(422, str(exc), kind="invalid") from None
    except IntegrityError:
        session.rollback()
        raise FeedbackError(409, "Knowledge request conflicts with current state; reload and retry",
                            kind="conflict") from None
    except OperationalError as exc:
        session.rollback()
        if getattr(exc.orig, "pgcode", None) in _BUSY_PGCODES:
            raise FeedbackError(409, "This run is being updated — try again in a moment", kind="busy") from None
        raise
    except BaseException:
        session.rollback()
        raise


def _scoped_case(session: Session, scope: KnowledgeScope, case_id: str) -> store.KbCase:
    try:
        return store._scoped_case(session, scope, case_id, store.InvalidSet.load(session))
    except KnowledgeAccessDenied:
        session.rollback()
        raise _not_found() from None


def scoped_run(session: Session, scope: KnowledgeScope, run_id: str):
    """The Run, resolved to its stored project (the Run case's, else the Run row's)."""
    from ..models import Run

    run = session.get(Run, run_id)
    root = store.current_case(session, store.run_case_key(run_id))
    project_id = root.project_id if root is not None else (run.project_id if run is not None else None)
    if run is None or project_id is None or not scope.allows_project(project_id):
        session.rollback()
        raise _not_found()
    return run


# ---------------------------------------------------------------- start (done-only bootstrap)


def _start_tx(session: Session, run_id: str, pre: Optional[PreHash], lookup: Optional[Callable]):
    store.lock_run_and_cases(session, run_id, [])
    if not store._run_is_done(session, run_id):
        raise store.NotDelivered("not delivered: outcome recording opens when this run is done")
    live = store._locked_result_path(session, run_id)
    if not live or not os.path.isfile(live):
        raise store.KnowledgeConflict("No delivered file is on disk for this run — it can't be judged here")
    case = store.ensure_run_case(session, run_id)
    store.lock_run_and_cases(session, None, [case.id])
    kwargs = {"live_link_lookup": lookup} if lookup is not None else {}
    store.sync_run_case_version(session, case, hasher=memo_hasher(pre), invalid=store.InvalidSet.load(session),
                                **kwargs)
    latest = store.latest_version(session, case.id)
    return {"case_id": case.id, "version_no": latest.version_no if latest is not None else None}


def start_run(session: Session, scope: KnowledgeScope, run_id: str, *,
              lookup: Optional[Callable] = None) -> dict:
    """State-idempotent: a replay finds the case and the same sha, and appends nothing."""
    run = scoped_run(session, scope, run_id)
    path = run.result_local_path
    session.rollback()  # end the read transaction; hash with no snapshot or lock held
    pre = prehash(path)
    return _run(session, _start_tx, run_id, pre, lookup if lookup is not None else live_link_lookup)


# ---------------------------------------------------------------- writes


def _verified_hasher(session: Session, case: store.KbCase, historical: bool):
    """Pre-hash outside the transaction (S10).

    Run case: the live final, handed to the store's sync as a memo hasher.
    Any other case judged as current: its one version must be a preserved
    locator, or its ``local_path`` bytes must still match the recorded sha.
    """
    from ..models import Run

    if case.scope == "run":
        run = session.get(Run, case.run_id)
        path = run.result_local_path if run is not None else None
        session.rollback()
        return memo_hasher(prehash(path))
    if historical:
        session.rollback()
        return store.sha256_file
    latest = store.latest_version(session, case.id)
    artifact = session.get(store.KbArtifact, latest.artifact_id) if latest is not None else None
    facts = (artifact.locator_kind, artifact.locator, artifact.sha256, latest.version_no) if artifact else None
    session.rollback()
    if facts is not None and facts[0] == "local_path":
        pre = prehash(facts[1])
        if facts[2] is None or pre is None or pre.sha256 != facts[2]:
            raise FeedbackError(409, "This output changed on disk since it was recorded", kind="conflict",
                                current_version_no=facts[3])
    return store.sha256_file


def submit_feedback(session: Session, scope: KnowledgeScope, case_id: str, *, idempotency_key: str,
                    expected_version_no: int, verdict: Optional[str], defects: list, note: Optional[str],
                    historical: bool, actor_label: Optional[str]) -> dict:
    case = _scoped_case(session, scope, case_id)
    hasher = _verified_hasher(session, case, historical)
    receipt = _run(session, store.submit_feedback, scope, case_id=case_id, idempotency_key=idempotency_key,
                   expected_version_no=expected_version_no, verdict=verdict, defects=defects, note=note,
                   historical=historical, actor_label=actor_label, hasher=hasher, require_done=True)
    return _receipt_out(receipt)


def record_correction(session: Session, scope: KnowledgeScope, case_id: str, **fields) -> dict:
    _scoped_case(session, scope, case_id)
    receipt = _run(session, store.record_correction, scope, case_id=case_id, actor_kind="operator", **fields)
    return _receipt_out(receipt)


def retract(session: Session, scope: KnowledgeScope, event_id: str, *, idempotency_key: str, reason: str,
            actor_label: Optional[str]) -> dict:
    receipt = _run(session, store.retract_event, scope, event_id=event_id, idempotency_key=idempotency_key,
                   reason=reason, actor_kind="operator", actor_label=actor_label)
    return _receipt_out(receipt)


def _receipt_out(receipt: store.Receipt) -> dict:
    return {"request_id": receipt.request_id, "kind": receipt.kind, "case_id": receipt.case_id,
            "event_ids": list(receipt.ids), "replayed": receipt.replayed}


# ---------------------------------------------------------------- external repair


def drive_file_id(link: str) -> str:
    """Parse to a Drive file id only — nothing is downloaded or hashed (R20)."""
    from ..gdrive_client import GDriveError, extract_file_id

    try:
        return extract_file_id(link)
    except (GDriveError, ValueError, TypeError):
        raise FeedbackError(422, "Not a Google Drive file link", kind="invalid") from None


def _disk_ok(root: str, nbytes: int) -> bool:
    usage = shutil.disk_usage(root)
    return usage.free - nbytes >= max(MIN_FREE_BYTES, MIN_FREE_FRACTION * usage.total)


def knowledge_root() -> str:
    from ..storage import data_dir_root

    return os.path.join(data_dir_root(), "knowledge")


def save_upload(fileobj, filename: str, project_id: str, size_hint: Optional[int] = None) -> dict:
    """Stream an operator upload to ``external/<project>/<sha>.<ext>``, hashing as it goes.

    Done before the write transaction (no lock is held while bytes move). The
    disk guard runs first (507), then the 500 MB cap (413), then ffprobe (422).
    """
    from .. import media

    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise FeedbackError(422, "Upload an .mp4, .mov or .m4v video", kind="invalid")
    if size_hint is not None and size_hint > UPLOAD_MAX_BYTES:
        raise FeedbackError(413, "The upload is larger than 500 MB — use a Drive link instead", kind="too_large")
    root = knowledge_root()
    tmp_dir = os.path.join(root, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    if not _disk_ok(tmp_dir, size_hint or UPLOAD_MAX_BYTES):
        raise FeedbackError(507, "Not enough disk space to keep this file — use a Drive link instead", kind="disk")
    tmp = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.part{ext}")
    digest, written = hashlib.sha256(), 0
    try:
        with open(tmp, "wb") as out:
            for chunk in iter(lambda: fileobj.read(1 << 20), b""):
                written += len(chunk)
                if written > UPLOAD_MAX_BYTES:
                    raise FeedbackError(413, "The upload is larger than 500 MB — use a Drive link instead",
                                        kind="too_large")
                digest.update(chunk)
                out.write(chunk)
        if written == 0:
            raise FeedbackError(422, "The uploaded file is empty", kind="invalid")
        try:
            info = media.probe(tmp)
        except Exception:
            raise FeedbackError(422, "The uploaded file is not a readable video", kind="invalid") from None
        sha = digest.hexdigest()
        dest_dir = os.path.join(root, "external", project_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{sha}{ext}")
        if os.path.exists(dest):
            os.unlink(tmp)  # content-addressed: the same bytes are already kept
        else:
            os.replace(tmp, dest)
        return {"locator": dest, "sha256": sha, "bytes": written,
                "duration_sec": getattr(info, "duration_sec", None)}
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def link_external_repair(session: Session, scope: KnowledgeScope, run_id: str, *, idempotency_key: str,
                         drive_link: Optional[str] = None, upload=None, upload_name: Optional[str] = None,
                         upload_size: Optional[int] = None, replaced_delivered: Optional[bool] = None,
                         note: Optional[str] = None, actor_label: Optional[str] = None) -> dict:
    if (drive_link is None) == (upload is None):
        raise FeedbackError(422, "Give either a Drive link or an upload", kind="invalid")
    run = scoped_run(session, scope, run_id)  # scope before any bytes are written
    project_id = run.project_id
    session.rollback()
    if drive_link is not None:
        media_fields = {"locator_kind": "drive_file", "locator": drive_file_id(drive_link)}
    else:
        saved = save_upload(upload, upload_name or "", project_id, upload_size)
        media_fields = {"locator_kind": "local_path", **saved}
    receipt = _run(session, store.record_external_repair, scope, run_id=run_id, idempotency_key=idempotency_key,
                   replaced_delivered=replaced_delivered, note=note, actor_label=actor_label, **media_fields)
    out = _receipt_out(receipt)
    external = store.external_case_for_event(session, receipt.ids[0]) if receipt.ids else None
    out["external_case_id"] = external.id if external is not None else None
    session.rollback()
    return out
