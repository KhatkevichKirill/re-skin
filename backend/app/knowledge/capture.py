"""Live capture hooks (AICP-299; docs/knowledge-capture-hooks.md).

Everything here is a no-op unless ``KNOWLEDGE_CAPTURE_ENABLED`` is set, the
database is PostgreSQL, and the project is in the explicit capture scope
(``KNOWLEDGE_PROJECT_IDS``; ``*`` is refused). With capture off, the pipeline
behaves byte-for-byte as before.

Rules every hook follows (design §0):

- P1: nothing is captured between a paid provider call and the pipeline's
  commit of that call's id. Submissions, outputs and terminal states are
  captured after the pipeline's own commit.
- P2: every DB-touching call runs on the capture executor (``capture_exec``)
  with a bounded wait; it never raises into the pipeline and never queues.
- P3: inside pipeline functions the only capture work is a hardlink
  (``link_preserve``). Publication links (``link_publication``) run in the
  caller after the pipeline function returned its ``PublishedFinal``.

All functions below are safe to call with any arguments: they return None
(or do nothing) on any problem.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from . import capture_exec
from .publication import REASON_FOR_KIND, PublishedFinal

log = logging.getLogger(__name__)

SITE_BUDGET_SEC = 3.0
LINK_FAILURE_BUDGET_SEC = 2.0
DEADLINE_MARGIN_SEC = 90.0

# Case creation (§1.4): server-side bounds on the Run-row hold.
CASE_LOCK_SQL = ("SET LOCAL lock_timeout = '200ms'", "SET LOCAL statement_timeout = '250ms'",
                 "SET LOCAL idle_in_transaction_session_timeout = '500ms'")

_DEADLINE: Optional[float] = None
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# ---------------------------------------------------------------- scope


@dataclass(frozen=True)
class CaptureScope:
    """What is recorded: an explicit list of project UUIDs (design §5)."""
    project_ids: frozenset
    valid: bool
    reason: str = ""

    @property
    def sha256(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(json.dumps(sorted(self.project_ids)).encode()).hexdigest()

    @property
    def scope8(self) -> str:
        return self.sha256[:8] if self.valid else ""

    @classmethod
    def parse(cls, configured) -> "CaptureScope":
        raw = configured if isinstance(configured, str) else ""
        entries = [e.strip().lower() for e in raw.split(",") if e.strip()]
        if not entries:
            return cls(frozenset(), False, "empty")
        if "*" in entries:
            return cls(frozenset(), False, "'*' is refused for capture; list the project UUIDs")
        if any(not _UUID.match(e) for e in entries):
            return cls(frozenset(), False, "invalid project id in KNOWLEDGE_PROJECT_IDS")
        return cls(frozenset(entries), True)

    def allows(self, project_id) -> bool:
        return self.valid and isinstance(project_id, str) and project_id.lower() in self.project_ids


_scope_cache: dict = {}


def capture_scope() -> CaptureScope:
    from ..config import settings

    raw = getattr(settings, "KNOWLEDGE_PROJECT_IDS", "")
    cached = _scope_cache.get("v")
    if cached is None or cached[0] != raw:
        scope = CaptureScope.parse(raw)
        if not scope.valid and getattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", False) and scope.reason != "empty":
            log.warning("KNOWLEDGE_CAPTURE scope refused: %s", scope.reason)
        cached = (raw, scope)
        _scope_cache["v"] = cached
    return cached[1]


def enabled() -> bool:
    """Flag on, PostgreSQL, and a valid explicit scope (no I/O)."""
    from ..config import settings

    if getattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", False) is not True:
        return False
    if not str(getattr(settings, "DATABASE_URL", "")).startswith("postgresql"):
        return False
    return capture_scope().valid


def in_scope(project_id) -> bool:
    return enabled() and capture_scope().allows(project_id)


def boot_line(process: str) -> str:
    from ..config import settings

    scope = capture_scope()
    return (f"KNOWLEDGE_CAPTURE process={process} enabled={enabled()} "
            f"preserve={bool(getattr(settings, 'KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA', False))} "
            f"capture_scope={len(scope.project_ids) if scope.valid else 'NONE'}")


# ---------------------------------------------------------------- deadline / budget


def begin_job() -> None:
    """First statement of every RQ wrapper that can reach a capture hook."""
    global _DEADLINE
    try:
        if not enabled():
            return
        from rq import get_current_job

        job = get_current_job()
        timeout = getattr(job, "timeout", None) if job is not None else None
        _DEADLINE = time.monotonic() + float(timeout) if timeout else None
    except Exception:
        _DEADLINE = None


def _budget(site_budget: float) -> Optional[float]:
    """The wait a capture call may impose here, or None to skip (§1.3)."""
    if _DEADLINE is not None:
        remaining = _DEADLINE - time.monotonic() - DEADLINE_MARGIN_SEC
        b = min(site_budget, remaining)
        return b if b >= 1.0 else None
    if threading.current_thread() is not threading.main_thread():
        return None  # a pool thread in a job that never called begin_job
    try:
        from rq import get_current_job

        if get_current_job() is not None:
            return None  # an RQ job whose wrapper forgot begin_job: skip, never risk its alarm
    except Exception:
        pass
    return site_budget


def _run(site: str, fn: Callable, budget: Optional[float] = None):
    b = _budget(SITE_BUDGET_SEC) if budget is None else budget
    if b is None:
        capture_exec.count(site, "deadline_skip")
        return None
    return capture_exec.executor().call(fn, b, site=site)


def _code_revision() -> Optional[str]:
    return os.environ.get("RESKIN_CODE_REVISION") or None


# ---------------------------------------------------------------- case creation (§1.4)


class _RunLocked(Exception):
    pass


def _segment_entry(run, segment_def_id):
    return next((e for e in (run.segment_snapshot or []) if isinstance(e, dict)
                 and e.get("id") == segment_def_id), None)


def _ensure_cases(session, run, rs=None):
    """(run_case, segment_case|None). Existing cases need no Run lock; a missing
    one is created in a short transaction under ``runs → kb_cases`` with
    server-side bounds, and on contention the call is skipped (counted)."""
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from . import store

    run_case = store.current_case(session, store.run_case_key(run.id))
    seg_case = store.current_case(session, store.segment_case_key(run.id, rs.segment_def_id)) \
        if rs is not None else None
    if run_case is not None and (rs is None or seg_case is not None):
        return run_case, seg_case
    session.commit()  # end the read transaction; the next one carries the bounds
    try:
        for sql in CASE_LOCK_SQL:
            session.execute(text(sql))
        store.lock_run_and_cases(session, run.id, [])
        run_case = store.ensure_run_case(session, run.id)
        if run_case is None:
            session.rollback()
            return None, None
        if rs is not None:
            entry = _segment_entry(run, rs.segment_def_id)
            start = end = None
            missing = []
            if entry is not None and isinstance(entry.get("start_sec"), (int, float)) \
                    and isinstance(entry.get("end_sec"), (int, float)) and 0 <= entry["start_sec"] < entry["end_sec"]:
                start, end = float(entry["start_sec"]), float(entry["end_sec"])
            else:
                missing.append("time_range")
            seg_case = store.ensure_case(
                session, case_key=store.segment_case_key(run.id, rs.segment_def_id), scope="segment",
                operation=run_case.operation, project_id=run.project_id, run_id=run.id, run_segment_id=rs.id,
                segment_def_id=rs.segment_def_id, parent_case_id=run_case.id, language=run.language,
                time_start_sec=start, time_end_sec=end, snapshot={"index": rs.index, "source": "capture"},
                missing_facts=missing)
        session.commit()
        return run_case, seg_case
    except OperationalError:
        session.rollback()
        capture_exec.count("case", "capture_skipped_run_locked")
        raise _RunLocked()


# ---------------------------------------------------------------- shared queries


def _attempt_for_task(session, task_id):
    """The valid live/import attempt that holds this provider task id, or None."""
    from sqlalchemy import select

    from . import store
    from .models import KbAttempt, KbAttemptSubmission

    invalid = store.InvalidSet.load(session)
    for sub, attempt in session.execute(
            select(KbAttemptSubmission, KbAttempt).join(KbAttempt, KbAttempt.id == KbAttemptSubmission.attempt_id)
            .where(KbAttemptSubmission.provider_task_id == task_id)).all():
        if not invalid.has("submission", sub.id) and not invalid.attempt_invalid(attempt):
            return attempt
    return None


def _latest_submitted(session, target_kind, target_id):
    """(attempt, task_id) of the newest valid submitted attempt on a target (S2)."""
    from sqlalchemy import select

    from . import store
    from .models import KbAttempt, KbAttemptSubmission

    invalid = store.InvalidSet.load(session)
    for sub, attempt in session.execute(
            select(KbAttemptSubmission, KbAttempt).join(KbAttempt, KbAttempt.id == KbAttemptSubmission.attempt_id)
            .where(KbAttempt.target_kind == target_kind, KbAttempt.target_id == target_id)
            .order_by(KbAttempt.seq.desc())).all():
        if not invalid.has("submission", sub.id) and not invalid.attempt_invalid(attempt):
            return attempt, sub.provider_task_id
    return None, None


def _observe(session, attempt_id, *, kind, dedupe=True, **kw):
    """S7: one observation per (attempt, kind, value), serialized per attempt."""
    from sqlalchemy import select

    from . import store
    from .models import KbAttemptObservation

    if dedupe:
        store._advisory_lock(session, "kb_attempt_obs", attempt_id)
        invalid = store.InvalidSet.load(session)
        rows = [o for o in session.execute(select(KbAttemptObservation).where(
            KbAttemptObservation.attempt_id == attempt_id, KbAttemptObservation.kind == kind)
            .order_by(KbAttemptObservation.seq.desc())).scalars() if not invalid.row_invalid("observation", o)]
        latest = rows[0] if rows else None
        if latest is not None:
            if kind == "status" and latest.status == kw.get("status"):
                return latest
            if kind == "output" and any(o.output_artifact_id == kw.get("output_artifact_id") for o in rows):
                return latest
            if kind == "error" and latest.error_class == kw.get("error_class") and \
                    (latest.payload or {}).get("stage") == (kw.get("payload") or {}).get("stage"):
                return latest
            if kind == "ambiguous":
                return latest
    return store.record_observation(session, attempt_id, kind=kind, **kw)


# ---------------------------------------------------------------- dispatch basis (§1.5)

_DRIVE_ID = re.compile(r"(?:/d/|[?&]id=)([A-Za-z0-9_-]{10,})")


def normalize_locator(item) -> Optional[str]:
    if not isinstance(item, str) or not item.strip():
        return None
    item = item.strip()
    if "drive.google.com" in item or "docs.google.com" in item:
        m = _DRIVE_ID.search(item)
        return f"drive:{m.group(1)}" if m else item
    if item.startswith(("http://", "https://")):
        return item
    try:
        from ..storage import data_dir_root

        real = os.path.realpath(item)
        root = data_dir_root()
        if real == root or real.startswith(root + os.sep):
            return "local:" + os.path.relpath(real, root)
    except Exception:
        pass
    return "local:" + item


def strip_volatile(obj):
    """The request without uploaded URLs (they change on every re-upload)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "url" or k.endswith("_url"):
                continue
            if k.endswith("_urls"):
                out[k] = len(v) if isinstance(v, list) else None
                continue
            out[k] = strip_volatile(v)
        return out
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def segment_ref_locators(run, rs, override=None) -> list:
    """Same priority the submit phase resolves (pipeline_v2._build_submit_state)."""
    if override:
        src = list(override)
    elif rs is not None and rs.generated_reference_urls:
        src = list(rs.generated_reference_urls)
    elif rs is not None and rs.reference_image_urls_override:
        src = list(rs.reference_image_urls_override)
    elif run.preprocess_enabled and run.preprocessed_reference_urls:
        src = list(run.preprocessed_reference_urls)
    else:
        src = list(run.reference_image_urls or [])
    return [normalize_locator(x) for x in src]


def _model_fields(run_model):
    from .. import ai_models

    try:
        spec = ai_models.spec_for(run_model)
        return run_model, spec.family
    except Exception:
        return run_model, None


def _swap_request(recipe: dict) -> dict:
    from ..kie_client import build_omni_payload, build_seedance_payload
    from ..pipeline_v2 import _swap_task_call

    method, kwargs = _swap_task_call(**recipe)
    return build_omni_payload(**kwargs) if method == "create_omni_task" else build_seedance_payload(**kwargs)


# ---------------------------------------------------------------- A: intents and submissions


def swap_intent(*, rs_id: str, recipe: dict, resubmit_of: Optional[str] = None,
                include_audio: Optional[bool] = None, attempt_kind: Optional[str] = None,
                origin: str = "automatic", target_kind: str = "run_segment", target_id: Optional[str] = None,
                idempotency_key: Optional[str] = None, ref_override=None, site: str = "A1") -> Optional[str]:
    """Record the intent of a swap/range video dispatch; returns the attempt id."""
    if not enabled():
        return None
    recipe = dict(recipe)

    def body(session):
        from .. import models as m
        from . import store
        from .store import KnowledgeConflict

        rs = session.get(m.RunSegment, rs_id)
        run = session.get(m.Run, rs.run_id) if rs is not None else None
        if run is None or not in_scope(run.project_id):
            return None
        try:
            _, seg_case = _ensure_cases(session, run, rs)
        except _RunLocked:
            return None
        if seg_case is None:
            return None
        request = _swap_request(recipe)
        kind, predecessor, parent = attempt_kind, None, None
        tgt = target_id or rs.id
        if resubmit_of:
            kind, predecessor = "resubmit", resubmit_of
            parent_attempt = _attempt_for_task(session, resubmit_of)
            parent = parent_attempt.id if parent_attempt is not None else None
        elif kind is None:
            latest, task = _latest_submitted(session, target_kind, tgt)
            kind, predecessor = ("rerun", task) if latest is not None else ("generation", None)
        basis = {"request": strip_volatile(request),
                 "ref_locators": segment_ref_locators(run, rs, ref_override),
                 "clip_locator": {"project_id": run.project_id, "clip_start": recipe.get("clip_start"),
                                  "clip_end": recipe.get("clip_end"), "include_audio": include_audio},
                 "voice_locators": [normalize_locator(x) for x in (run.reference_audio_urls or [])]}
        model_key, family = _model_fields(recipe.get("run_model"))
        ctx = {"site": site, "segment_def_id": rs.segment_def_id, "index": rs.index,
               "clip_start": recipe.get("clip_start"), "clip_end": recipe.get("clip_end"),
               "generate_audio": recipe.get("generate_audio"), "include_audio": include_audio,
               "dispatch_basis": basis, "capture_scope_sha256": capture_scope().sha256}
        args = dict(case_id=seg_case.id, attempt_kind=kind, origin=origin, provider="kie", request=request,
                    target_kind=target_kind, target_id=tgt, predecessor_task_id=predecessor,
                    model_key=model_key, provider_model_id=request.get("model"), model_family=family,
                    recipe_context=ctx, code_revision=_code_revision(), parent_attempt_id=parent,
                    dispatch_basis=basis, redispatch_payload={"site": site, "pid": os.getpid()})
        try:
            attempt, _ = store.record_attempt_intent(session, idempotency_key=idempotency_key, **args)
        except KnowledgeConflict:
            if idempotency_key is None:
                raise
            session.rollback()  # a withdrawn keyed attempt: record under a fresh key
            attempt, _ = store.record_attempt_intent(session, idempotency_key=None, **args)
        return attempt.id

    return _run(site, body)


def image_intent(*, run_id: str, rs_id: Optional[str], target_kind: str, target_id: str, attempt_kind: str,
                 origin: str, prompt: str, input_urls: list, locators: Iterable, recipe_context: dict,
                 site: str) -> Optional[str]:
    """Intent of a GPT Image dispatch (run preprocess or segment image job)."""
    if not enabled():
        return None
    input_urls, locators, recipe_context = list(input_urls), list(locators), dict(recipe_context)

    def body(session):
        return _image_intent_body(session, run_id=run_id, rs_id=rs_id, target_kind=target_kind,
                                  target_id=target_id, attempt_kind=attempt_kind, origin=origin, prompt=prompt,
                                  input_urls=input_urls, locators=locators, recipe_context=recipe_context,
                                  site=site)

    return _run(site, body)


def _image_intent_body(session, *, run_id, rs_id, target_kind, target_id, attempt_kind, origin, prompt,
                       input_urls, locators, recipe_context, site):
    from .. import models as m
    from ..kie_client import build_image_payload
    from . import store

    run = session.get(m.Run, run_id)
    if run is None or not in_scope(run.project_id):
        return None
    rs = session.get(m.RunSegment, rs_id) if rs_id else None
    try:
        run_case, seg_case = _ensure_cases(session, run, rs)
    except _RunLocked:
        return None
    case = seg_case or run_case
    if case is None:
        return None
    request = build_image_payload(prompt=prompt, input_urls=input_urls)
    basis = {"request": strip_volatile(request), "ref_locators": [normalize_locator(x) for x in locators]}
    ctx = dict(recipe_context, site=site, dispatch_basis=basis, capture_scope_sha256=capture_scope().sha256)
    attempt, _ = store.record_attempt_intent(
        session, case_id=case.id, attempt_kind=attempt_kind, origin=origin, provider="kie", request=request,
        target_kind=target_kind, target_id=target_id, model_key=request["model"],
        provider_model_id=request["model"], model_family="gpt_image", recipe_context=ctx,
        code_revision=_code_revision(), dispatch_basis=basis, redispatch_payload={"site": site, "pid": os.getpid()})
    return attempt.id


def submitted(attempt_id: Optional[str], task_id: Optional[str], *, predecessor_failed: Optional[dict] = None,
              site: str = "A") -> None:
    """After the pipeline's own commit of ``task_id`` (P1)."""
    if not attempt_id or not task_id or not enabled():
        return

    def body(session):
        from . import store

        store.record_submission(session, attempt_id, provider_task_id=task_id, spend_call_key=task_id)
        if predecessor_failed and predecessor_failed.get("task_id"):
            prev = _attempt_for_task(session, predecessor_failed["task_id"])
            if prev is not None:
                _observe(session, prev.id, kind="status", status="failed",
                         error_class=_short(predecessor_failed.get("error_class")),
                         payload=predecessor_failed.get("payload"))
        return True

    _run(site, body)


def dispatch_error(attempt_id: Optional[str], exc: BaseException, site: str = "A") -> None:
    """The paid call raised: the intent stays open (acceptance unknown)."""
    if not attempt_id or not enabled():
        return
    err, msg = type(exc).__name__, str(exc)[:500]

    def body(session):
        return _observe(session, attempt_id, kind="error", error_class=_short(err),
                        payload={"stage": "dispatch", "message": msg})

    _run(site, body)


def intent_and_submission_batch(*, run_id: str, items: list, site: str = "A5") -> None:
    """A5: per_image preprocess tasks, recorded together after the loop's commit.

    ``items``: dicts with ``index``, ``task_id``, ``prompt``, ``input_urls``, ``locators``.
    """
    if not items or not enabled():
        return
    items = [dict(i) for i in items]

    def body(session):
        from . import store

        for it in items:
            aid = _image_intent_body(session, run_id=run_id, rs_id=None, target_kind="run_preprocess",
                                     target_id=f"{run_id}:{it['index']}", attempt_kind="preprocess_image",
                                     origin="automatic", prompt=it["prompt"], input_urls=it["input_urls"],
                                     locators=it["locators"],
                                     recipe_context={"ref_index": it["index"], "mode": "per_image",
                                                     "intent_recorded_after_call": True}, site=site)
            if aid is None:
                return None
            store.record_submission(session, aid, provider_task_id=it["task_id"], spend_call_key=it["task_id"])
        return True

    _run(site, body)


def range_ambiguous(attempt_id: str, billed_at) -> None:
    """A3: the next claim found an ambiguous prior submission."""
    if not enabled():
        return

    def body(session):
        from sqlalchemy import select

        from .models import KbAttempt

        a = session.execute(select(KbAttempt).where(KbAttempt.idempotency_key == f"range_attempt:{attempt_id}")
                            ).scalar_one_or_none()
        if a is None:
            return None
        _observe(session, a.id, kind="ambiguous", payload={"reason": "submit_billed_without_task_id",
                                                           "billed_at": str(billed_at)})
        _observe(session, a.id, kind="status", status="unknown")
        return True

    _run("A3", body)


# ---------------------------------------------------------------- B / T: outputs and terminal states


def output(*, task_id: str, path: str, run_id: str, rs_id: Optional[str], status: str = "succeeded",
           payload: Optional[dict] = None, artifact_kind: str = "model_output", segment_version: bool = False,
           attempt_kind: str = "generation", origin: str = "automatic", target_kind: str = "run_segment",
           target_id: Optional[str] = None, site: str = "B") -> None:
    """A downloaded provider result, recorded after the pipeline's commit."""
    if not task_id or not path or not enabled():
        return
    payload = dict(payload or {})

    def body(session):
        from .. import models as m
        from ..config import settings
        from . import store
        from .models import KbArtifact

        run = session.get(m.Run, run_id)
        if run is None or not in_scope(run.project_id):
            return None
        rs = session.get(m.RunSegment, rs_id) if rs_id else None
        try:
            run_case, seg_case = _ensure_cases(session, run, rs)
        except _RunLocked:
            return None
        case = seg_case or run_case
        if case is None or not os.path.isfile(path):
            return None
        attempt = _attempt_for_task(session, task_id)
        if attempt is None:  # orphan: output without a recorded intent
            attempt, _ = store.record_attempt_intent(
                session, case_id=case.id, attempt_kind=attempt_kind, origin=origin, provider="kie", request=None,
                target_kind=target_kind, target_id=target_id or rs_id or run_id,
                recipe_context={"orphan": True, "capture_scope_sha256": capture_scope().sha256},
                idempotency_key=f"orphan:kie:{task_id}")
            store.record_submission(session, attempt.id, provider_task_id=task_id, spend_call_key=task_id)
            _observe(session, attempt.id, kind="ambiguous", payload={"reason": "output_without_intent"})
        sha = store.sha256_file(path)
        locator_kind, locator = "local_path", path
        if getattr(settings, "KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA", False):
            kept = _preserve_attempt_media(path, run.project_id, attempt.id)
            if kept:
                locator_kind, locator = "preserved_copy", kept
        art = store.get_or_create_artifact(session, kind=artifact_kind, locator_kind=locator_kind, locator=locator,
                                           sha256=sha, project_id=run.project_id, bytes=os.path.getsize(path))
        _observe(session, attempt.id, kind="output", output_artifact_id=art.id, payload={"pipeline_path": path})
        _observe(session, attempt.id, kind="status", status=status, payload=payload or None)
        if segment_version and seg_case is not None:
            store.lock_run_and_cases(session, None, [seg_case.id])
            invalid = store.InvalidSet.load(session)
            latest = store.latest_version(session, seg_case.id, invalid)
            if latest is None or session.get(KbArtifact, latest.artifact_id).sha256 != sha:
                # The importer's record for one provider task's segment output, so
                # live capture and import name the same publication (no duplicates).
                store.append_case_version(session, seg_case, artifact_id=art.id,
                                          reason="initial_output" if latest is None else "regeneration",
                                          source_record=f"import:run_segments:{rs.id}:{task_id}",
                                          invalid=invalid)
        return True

    _run(site, body)


def _preserve_attempt_media(path: str, project_id: str, attempt_id: str) -> Optional[str]:
    from .links import knowledge_root

    ext = os.path.splitext(path)[1] or ".bin"
    d = os.path.join(knowledge_root(), "attempts", project_id)
    target = os.path.join(d, f"{attempt_id}{ext}")
    try:
        os.makedirs(d, exist_ok=True)
        if os.path.exists(target):
            return target if os.path.samefile(target, path) else None
        os.link(path, target)  # same filesystem: no data blocks copied
        return target
    except OSError:
        capture_exec.count("attempt_media", "preserve_failed")
        return None


def terminal(*, task_id: Optional[str], status: str, error_class: Optional[str] = None,
             payload: Optional[dict] = None, site: str = "T") -> None:
    """A provider/pipeline terminal state for a recorded attempt (no orphan path)."""
    if not task_id or not enabled():
        return
    payload = dict(payload or {})

    def body(session):
        attempt = _attempt_for_task(session, task_id)
        if attempt is None:
            return None
        _observe(session, attempt.id, kind="status", status=status, error_class=_short(error_class),
                 payload=payload or None)
        return True

    _run(site, body)


def _short(value) -> Optional[str]:
    return None if value is None else str(value)[:64]


# ---------------------------------------------------------------- finals: links (§3.2)


@dataclass
class LinkFailed:
    publication: PublishedFinal
    exc: BaseException
    stat: Optional[os.stat_result]
    failed_at_ns: int


def link_publication(pf: Optional[PublishedFinal]) -> Optional[LinkFailed]:
    """Point (i): the last statement after all committed work. Never raises —
    it absorbs RQ's timeout too (design §3.2) — and returns the failure, if any,
    for ``record_link_failure`` after the lock release."""
    if pf is None or not in_scope(pf.project_id):
        return None
    st = None
    try:
        from . import links

        st = os.stat(pf.path)
        links.make_link(pf.path, pf.project_id, pf.run_id, pf.kind, pf.publication_id,
                        capture_scope().scope8, st)
        return None
    except Exception as exc:  # includes rq.timeouts.BaseTimeoutException (an Exception subclass)
        capture_exec.count("link", "link_failed")
        log.warning("KNOWLEDGE_CAPTURE site=link outcome=link_failed publication=%s err=%s: %s",
                    pf.publication_id, type(exc).__name__, exc)
        return LinkFailed(pf, exc, st, time.time_ns())


def link_preserve(run_id: str, project_id: str, path: str) -> None:
    """Point (ii): before a stitch replaces ``final.mp4``. Re-raises RQ's
    timeout (the stitch would have hit it next); swallows everything else."""
    if not in_scope(project_id):
        return
    try:
        from rq.timeouts import BaseTimeoutException
    except Exception:  # pragma: no cover
        BaseTimeoutException = ()  # noqa: N806
    try:
        from . import links

        st = os.stat(path)
        if st.st_nlink != 1:
            return  # point (i) or an earlier (ii) already holds this inode
        links.make_link(path, project_id, run_id, links.SUPERSEDED,
                        f"superseded:{run_id}:{st.st_dev}-{st.st_ino}-{st.st_mtime_ns}", capture_scope().scope8, st)
    except FileNotFoundError:
        return
    except BaseTimeoutException:
        raise
    except Exception as exc:
        capture_exec.count("link_preserve", "link_failed")
        log.warning("KNOWLEDGE_CAPTURE site=link_preserve outcome=link_failed run=%s err=%s: %s",
                    run_id, type(exc).__name__, exc)


def record_link_failure(failed: Optional[LinkFailed]) -> None:
    """Durable ``kb_link_failures`` row, after the lock release (≤ 2 s or fire-and-forget)."""
    if failed is None:
        return
    pf, st = failed.publication, failed.stat
    values = dict(publication_id=pf.publication_id, run_id=pf.run_id, project_id=pf.project_id, kind=pf.kind,
                  path=pf.path, st_dev=st.st_dev if st else None, st_ino=st.st_ino if st else None,
                  st_mtime_ns=st.st_mtime_ns if st else None, failed_at_ns=failed.failed_at_ns,
                  error_class=_short(type(failed.exc).__name__))

    def body(session):
        import uuid

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from .models import KbLinkFailure

        session.execute(pg_insert(KbLinkFailure).values(id=str(uuid.uuid4()), **values)
                        .on_conflict_do_nothing(constraint="uq_kb_link_failure_publication"))
        return True

    try:
        b = _budget(LINK_FAILURE_BUDGET_SEC)
        ok = capture_exec.executor().call(body, b if b is not None else 0, site="link_failure")
        if not ok:
            capture_exec.count("link", "link_failure_unrecorded")
            log.warning("KNOWLEDGE_CAPTURE site=link outcome=link_failure_unrecorded publication=%s",
                        pf.publication_id)
    except Exception:
        pass


def record_capture_scope(session, recorded_by: str) -> None:
    """S13a: insert-or-get the active capture scope (provenance)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from .models import KbCaptureScope

    scope = capture_scope()
    if not scope.valid:
        return
    session.execute(pg_insert(KbCaptureScope).values(
        scope_sha256=scope.sha256, project_ids=sorted(scope.project_ids), recorded_by=recorded_by[:128])
        .on_conflict_do_nothing(index_elements=["scope_sha256"]))


__all__ = ["CaptureScope", "LinkFailed", "PublishedFinal", "REASON_FOR_KIND", "begin_job", "boot_line",
           "capture_scope", "dispatch_error", "enabled", "image_intent", "in_scope", "intent_and_submission_batch",
           "link_preserve", "link_publication", "normalize_locator", "output", "range_ambiguous",
           "record_capture_scope", "record_link_failure", "segment_ref_locators", "strip_volatile", "submitted",
           "swap_intent", "terminal"]
