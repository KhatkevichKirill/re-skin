"""Source adapters: existing production rows → knowledge store calls (design §2, §4.2, §6.3).

Each adapter maps one table, scanned in ``(created_at, id)`` order. What a row
does not record stays unknown: no adapter infers a verdict, a defect, a reason,
a provider request or a time range, and none ever writes an operator verdict.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import select

from .. import store
from ..models import KbArtifact, KbCase, KbCaseVersion
from .context import ImportContext, MediaChanged, model_provenance, operation_for, status_value



def _job_timeout_sec() -> int:
    return int(os.getenv("PROCESS_JOB_TIMEOUT", "10800"))


# Rows are scanned only once they are older than this. Every writer of these
# tables runs inside an RQ job (killed at PROCESS_JOB_TIMEOUT) or a short HTTP
# request, so a row whose created_at (or stamp) is older than twice the job
# timeout is either committed or never will be: a (ts, id) cursor below this
# horizon cannot step over a row that becomes visible later.
COMMIT_LAG = timedelta(seconds=2 * _job_timeout_sec())


def horizon(ctx, column) -> datetime:
    """``ctx.now - COMMIT_LAG`` in the column's own timezone convention."""
    edge = ctx.now - COMMIT_LAG
    return edge if getattr(column.type, "timezone", False) else edge.astimezone(timezone.utc).replace(tzinfo=None)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _terminal(status, terminal: tuple) -> bool:
    return status_value(status) in terminal


def _sha(value) -> Optional[str]:
    return value if isinstance(value, str) and _HEX64.match(value) else None


def _draw_for(ctx: ImportContext, call_key: Optional[str]):
    from ...models import KieSpendDraw

    if not call_key:
        return None
    return ctx.session.execute(select(KieSpendDraw).where(KieSpendDraw.call_key == call_key)).scalar_one_or_none()


def _terminal_status(ctx: ImportContext, attempt, status, *, succeeded: tuple, failed: tuple,
                     error: Optional[str] = None, output=None) -> None:
    value = status_value(status)
    if value in succeeded:
        if output is not None:
            ctx.observe(attempt, kind="output", output_artifact_id=output.id)
        ctx.observe(attempt, kind="status", status="succeeded")
    elif value in failed:
        ctx.observe(attempt, kind="status", status="failed",
                    payload={"error_message": (error or "")[:1000]} if error else None)


# ---------------------------------------------------------------- shared cases


def run_case(ctx: ImportContext, run) -> KbCase:
    """Run case from the live Run row, created under the Run lock (§6.2)."""
    from ...models import VideoProject

    before = run.result_local_path
    ctx.lock_run(run.id)
    if ctx.run_locked:
        # The Run was read before the lock: re-read it, and if a promotion moved
        # the final meanwhile, the prefetched media are stale (retry next pass).
        ctx.session.refresh(run)
        if run.result_local_path != before and ctx.media:
            raise MediaChanged(f"run {run.id} final moved while waiting for its lock")
    project = ctx.session.get(VideoProject, run.project_id)
    model_key, provider_model_id, family = model_provenance(status_value(run.model))
    missing = ["operator_verdict_history", "superseded_segment_attempts"]
    if run.segment_snapshot is None:
        missing.append("segment_snapshot")
    if not run.result_local_path:
        missing.append("result_local_path")
    if project is not None and operation_for(project) != status_value(project.project_type):
        missing.append("project_type")
    snapshot = {
        "run_name": run.name, "model_key": model_key, "provider_model_id": provider_model_id, "model_family": family,
        "resolution": status_value(run.resolution), "audio_mode": status_value(run.audio_mode),
        "status_at_import": status_value(run.status), "preprocess_enabled": bool(run.preprocess_enabled),
        "preprocess_mode": run.preprocess_mode, "reference_image_count": len(run.reference_image_urls or []),
        "reference_audio_count": len(run.reference_audio_urls or []),
        "stitch_plan": None if run.stitch_plan is None else ("none" if run.stitch_plan == [] else "custom"),
        "source_fallback_segments": run.source_fallback_segments, "result_gdrive_file_id": run.result_gdrive_file_id,
        "result_duration_sec": run.result_duration_sec, "mute_source": getattr(project, "mute_source", None),
        "max_segment_sec": getattr(project, "max_segment_sec", None), "source": "runs",
    }
    return ctx.case(case_key=store.run_case_key(run.id), scope="run", operation=operation_for(project),
                    project_id=run.project_id, run_id=run.id, language=run.language, snapshot=snapshot,
                    missing_facts=missing)


def segment_case(ctx: ImportContext, run, rs) -> KbCase:
    parent = run_case(ctx, run)
    entry = next((e for e in (run.segment_snapshot or []) if isinstance(e, dict)
                  and e.get("id") == rs.segment_def_id), None)
    start = end = None
    missing = []
    if entry is not None and isinstance(entry.get("start_sec"), (int, float)) \
            and isinstance(entry.get("end_sec"), (int, float)) and 0 <= entry["start_sec"] < entry["end_sec"]:
        start, end = float(entry["start_sec"]), float(entry["end_sec"])
    else:
        missing.append("time_range")
        if entry is None:
            ctx.report.note("missing_metadata", what="segment not in the run's segment_snapshot",
                            run_id=run.id, run_segment_id=rs.id)
    snapshot = {"index": rs.index, "status_at_import": status_value(rs.status),
                "source_fallback": bool(rs.source_fallback), "prompt_override_set": rs.prompt_override is not None,
                "reference_override_count": len(rs.reference_image_urls_override or []),
                "generated_reference_count": len(rs.generated_reference_urls or []),
                "generated_image_job_id": rs.generated_image_job_id, "source": "run_segments"}
    return ctx.case(case_key=store.segment_case_key(run.id, rs.segment_def_id), scope="segment",
                    operation=parent.operation, project_id=run.project_id, run_id=run.id, run_segment_id=rs.id,
                    segment_def_id=rs.segment_def_id, parent_case_id=parent.id, language=run.language,
                    time_start_sec=start, time_end_sec=end, snapshot=snapshot, missing_facts=missing)


def _same(a_path, a_sha, b_path, b_sha) -> bool:
    return a_sha == b_sha if a_sha and b_sha else a_path == b_path


def _publication(ctx: ImportContext, publication_id: str, mode: str, version_no: Optional[int]) -> None:
    items = ctx.report.extra.setdefault("publications", [])
    if len(items) < 200:
        items.append({"publication_id": publication_id, "mode": mode, "case_version_no": version_no})


def _matching_version(ctx: ImportContext, case: KbCase, path: Optional[str], sha: Optional[str],
                      below: Optional[int] = None, by_path: bool = False):
    """Newest valid version whose artifact has this content (sha, or path when
    unhashed), streamed newest-first from an indexed query: never the whole
    timeline under the Run lock, and never stopping before a valid match."""
    if sha is None and not path:
        return None
    if sha is not None:
        same = KbArtifact.sha256 == sha
    elif by_path:  # content unknown on our side: the recorded path names it (any hash)
        same = KbArtifact.locator == path
    else:
        same = KbArtifact.sha256.is_(None) & (KbArtifact.locator == path)
    q = (select(KbCaseVersion, KbArtifact).join(KbArtifact, KbArtifact.id == KbCaseVersion.artifact_id)
         .where(KbCaseVersion.case_id == case.id, same))
    if below is not None:
        q = q.where(KbCaseVersion.version_no < below)
    # Withdrawn rows excluded by indexed SQL predicates: one row, however long the
    # history and however large a withdrawn batch.
    q = q.where(~store.withdrawn_clause("case_version", KbCaseVersion.id),
                ~store.withdrawn_clause("artifact", KbArtifact.id))
    return ctx.session.execute(q.order_by(KbCaseVersion.version_no.desc()).limit(1)).first()


def _has_current_version(ctx: ImportContext, case: KbCase) -> bool:
    q = select(KbCaseVersion.id).where(KbCaseVersion.case_id == case.id, KbCaseVersion.historical.is_(False),
                                       ~store.withdrawn_clause("case_version", KbCaseVersion.id),
                                       ~store.withdrawn_clause("artifact", KbCaseVersion.artifact_id))
    return ctx.session.execute(q.limit(1)).first() is not None


def _publication_recorded(ctx: ImportContext, case: KbCase, publication_id: str) -> bool:
    return store.valid_publication_version(ctx.session, case.id, publication_id, ctx.invalid()) is not None


def promotion_version(ctx: ImportContext, case: KbCase, run, *, path: str, sha: Optional[str], reason: str,
                      publication_id: str, published_at=None):
    """Record what a promotion published: ``(version or None, artifact)``.

    * a version already appended for this publication → that version;
    * a version with the same content (appended by sync, reconciliation or
      live capture) → that version, mapped to this publication in the report
      and, by the caller, in the correction / attempt output that cites it;
    * still the Run's current final → appended (reason = the promotion);
    * superseded since → an S16 historical version: it takes the next
      number but never becomes current, and can be judged historically.

    Promotion outputs are new paths the pipeline never rewrites, so hashing
    one now is hashing what was published.
    """
    # Bounded lookups (indexed by publication / content), never the full history:
    # this runs under the Run lock.
    v = store.valid_publication_version(ctx.session, case.id, publication_id, ctx.invalid())
    if v is not None:
        return v, ctx.session.get(KbArtifact, v.artifact_id)
    if sha is None:
        sha, present = ctx.media_sha(path, what="promoted final")
        verified = sha is not None
    else:
        present, verified = ctx.present(path), False
    match = _matching_version(ctx, case, path, sha)
    if match is not None:
        v, art = match
        ctx.report.note("duplicates", what="promotion content already a version", publication_id=publication_id,
                        case_version_no=v.version_no)
        _publication(ctx, publication_id, "matched_existing_version", v.version_no)
        return v, art
    art = ctx.artifact(kind="stitched_final", locator_kind="local_path", locator=path, project_id=case.project_id,
                       sha256=sha, present=present, verified=verified, immutable_path=True)
    if run.result_local_path == path and status_value(run.status) == "done":
        version = ctx.append_version(case, art, reason=reason, source_record=publication_id)
        _publication(ctx, publication_id, "appended", version.version_no)
        return version, art
    version = ctx.append_version(case, art, reason=reason, source_record=publication_id, historical=True,
                                 published_at=published_at)
    ctx.report.note("historical", what="promotion superseded; recorded as a historical version",
                    run_id=run.id, publication_id=publication_id, case_version_no=version.version_no)
    _publication(ctx, publication_id, "historical_version", version.version_no)
    return version, art


def _correction(ctx: ImportContext, case: KbCase, *, table: str, row_id: str, before_path: Optional[str],
                before_sha: Optional[str], after_artifact: Optional[KbArtifact], after_version, method: str,
                publication_id: str) -> None:
    """A correction event only when both sides are recorded facts; resolved stays unknown.

    The base must be a recorded version earlier than the published one (any
    version when the publication is a historical artifact).
    """
    if after_artifact is None:
        return
    match = _matching_version(ctx, case, before_path, before_sha, by_path=True,
                              below=after_version.version_no if after_version is not None else None) \
        if before_path else None
    before = match[1] if match is not None else None
    if before is None:
        ctx.report.note("missing_metadata", what="correction base final is not a recorded version",
                        source=f"{table}:{row_id}")
        return
    key, done = ctx.request_key("import:correction:" + store.canonical_sha256([table, row_id, case.id])[:48])
    if done:
        ctx.report.note("duplicates", what="correction", source=f"{table}:{row_id}")
        return
    note = f"imported from {table}:{row_id}; publication {publication_id} -> " + (
        f"case version {after_version.version_no}" if after_version is not None else "historical artifact")
    receipt = store.record_correction(ctx.session, ctx.scope, case_id=case.id, idempotency_key=key,
                                      before_artifact_id=before.id, after_artifact_id=after_artifact.id,
                                      method=method, resolved="unknown", actor_kind="importer",
                                      note=note, import_batch_id=ctx.ensure_batch())
    ctx.register_events(receipt)


def record_export(ctx: ImportContext, case: KbCase, table: str, row) -> bool:
    """A Final Creatives stamp as its own ``export_observed`` event, once per stamp."""
    stamp = _utc(getattr(row, "final_creatives_exported_at", None))
    if stamp is None:
        return False
    if ctx.dry_run:
        # The writer takes the production Run row lock (§6.2); a dry run never
        # locks a Run, so the event is counted once, not written (it would roll back).
        seen = ctx.session.info.setdefault("kb_dry_exports", set())
        if (table, row.id, stamp) not in seen:
            seen.add((table, row.id, stamp))
            ctx.report.bump(ctx.report.extra.setdefault("reconcile", {}), "exports_observed_dry_run")
        return True
    if ctx.defer_exports:
        # Inside a source row the Run is already locked; the writer's order is
        # key → Run → case, so the event is written after the row commits.
        ctx.deferred_exports.append((case.id, table, row.id, stamp))
        return True
    return write_export(ctx, case.id, table, row.id, stamp)


def write_export(ctx: ImportContext, case_id: str, table: str, row_id: str, stamp: datetime) -> bool:
    """Called with no Run lock held: the store takes key → Run → case itself."""
    key, recorded = ctx.request_key("import:export:" + store.canonical_sha256(
        [table, row_id, case_id, stamp.isoformat()])[:48])
    if recorded:
        return False
    receipt = store.record_export_observed(ctx.session, case_id=case_id, idempotency_key=key, exported_at=stamp,
                                           source_record=f"{table}:{row_id}", import_batch_id=ctx.ensure_batch())
    ctx.register_events(receipt)
    ctx.report.bump(ctx.report.extra.setdefault("reconcile", {}), "exports_observed")
    return True


# ---------------------------------------------------------------- adapters


@dataclass(frozen=True)
class Source:
    name: str
    model: Callable[[], type]
    ready: Callable[[object, datetime], bool]
    process: Callable[[ImportContext, object], None]
    task_bearing: bool = False
    ages_out: bool = False
    media: Callable[[ImportContext, object], list] = lambda ctx, row: []  # (path, hash?) probed before any lock
    # Mutable rows (reset in place by reruns, retries, re-preprocessing) are scanned
    # by modification time so every change re-surfaces them; adapters are idempotent.
    cursor_col: str = "created_at"


CHAIN_MAX = 50  # bounds the runs adapter's work under the Run lock


def _chain_entries(ctx: ImportContext, run) -> list:
    """Recorded promotions of a Run: (when, before_path, before_sha, after_path, after_sha, reason, pub)."""
    from ...assistant_models import AssistantRunResultPromotion
    from ...models import RunAudioPatch, RunSegmentRangeAttempt

    cached = ctx.row_cache.get(("chain", run.id))
    if cached is not None:
        return cached  # computed by the media prefetch, before the Run lock
    entries = []
    # Runs once per row, during the media prefetch (before the Run lock), so even
    # a sort of a long promotion history never holds a production Run row.
    for p in ctx.session.execute(select(RunAudioPatch).where(RunAudioPatch.run_id == run.id,
                                                             RunAudioPatch.promoted_at.isnot(None))
                                 .order_by(RunAudioPatch.promoted_at.desc()).limit(CHAIN_MAX)).scalars():
        if p.output_path:
            entries.append((_utc(p.promoted_at), p.base_result_path, None, p.output_path, None, "audio_patch",
                            f"audio_patch:{p.id}"))
    for p in ctx.session.execute(select(AssistantRunResultPromotion).where(
            AssistantRunResultPromotion.run_id == run.id, AssistantRunResultPromotion.noop.is_(False))
            .order_by(AssistantRunResultPromotion.created_at.desc()).limit(CHAIN_MAX)).scalars():
        b, a = p.before_json or {}, p.after_json or {}
        if a.get("path"):
            entries.append((_utc(p.created_at), b.get("path"), _sha(b.get("sha256")), a["path"],
                            _sha(a.get("sha256")), "assistant_promotion", f"assistant_promotion:{p.id}"))
    for r in ctx.session.execute(select(RunSegmentRangeAttempt).where(
            RunSegmentRangeAttempt.run_id == run.id, RunSegmentRangeAttempt.promoted_at.isnot(None))
            .order_by(RunSegmentRangeAttempt.promoted_at.desc()).limit(CHAIN_MAX)).scalars():
        if r.final_local_path:
            entries.append((_utc(r.promoted_at), None, None, r.final_local_path, None, "range_attempt",
                            f"range_attempt:{r.id}"))
    # The rebuild runs under the Run lock: keep it bounded. Older promotions are
    # still recorded by their own adapters (as historical versions).
    entries.sort(key=lambda e: e[0])
    if len(entries) > CHAIN_MAX:
        ctx.report.note("missing_metadata", what=f"promotion chain truncated to the newest {CHAIN_MAX}",
                        run_id=run.id, dropped=len(entries) - CHAIN_MAX)
        entries = entries[-CHAIN_MAX:]
    ctx.row_cache[("chain", run.id)] = entries
    return entries


def _runs_media(ctx: ImportContext, run) -> list:
    """Only the current final is hashed; historical finals are probed for presence."""
    paths = [(run.result_local_path, True)] if run.result_local_path else []
    for _, b_path, _, a_path, _, _, _ in _chain_entries(ctx, run):
        paths += [(b_path, False), (a_path, False)]
    return paths


def _runs(ctx: ImportContext, run) -> None:
    case = run_case(ctx, run)
    # P3: only a delivered final is a Run-case version; a failed/incomplete run's
    # final.mp4 was never delivered (revisited until it is done).
    # Historical versions (a promotion imported alone, S16) never count as current.
    if status_value(run.status) == "done" and not _has_current_version(ctx, case):
        entries = _chain_entries(ctx, run)
        chain: list = []  # (path, sha, reason, source_record); historical files are never hashed now
        for _, b_path, b_sha, a_path, a_sha, reason, pub in sorted(entries, key=lambda e: e[0]):
            if b_path and (not chain or not _same(chain[-1][0], chain[-1][1], b_path, b_sha)):
                chain.append((b_path, b_sha, "import", f"import:before:{pub}"))
            chain.append((a_path, a_sha, reason, pub))
        current = run.result_local_path
        if current and (not chain or chain[-1][0] != current):
            chain.append((current, None, "import", f"import:runs:{run.id}"))
        for n, (path, sha, reason, source_record) in enumerate(chain):
            last = n == len(chain) - 1
            if _publication_recorded(ctx, case, source_record):
                continue  # that publication is already a (historical) version
            verified = False
            if sha is None and last and path == current:
                sha, present = ctx.media_sha(path, what="run final")
                verified = sha is not None
            else:
                present = ctx.present(path)
                if not present:
                    ctx.report.note("missing_media", what="historical run final", path=path)
            kind = "delivered_final" if last and run.result_gdrive_file_id else "stitched_final"
            art = ctx.artifact(kind=kind, locator_kind="local_path", locator=path, project_id=run.project_id,
                               sha256=sha, present=present, verified=verified, immutable_path=reason != "import",
                               duration_sec=run.result_duration_sec if last else None)
            ctx.append_version(case, art, reason=reason, source_record=source_record)
    record_export(ctx, case, "runs", run)
    model_key = None  # run-level preprocessing calls GPT Image, not the run's video model
    for i, task_id in enumerate(run.preprocess_task_ids or []):  # one per reference image: small by construction
        if not isinstance(task_id, str) or not task_id:
            continue
        attempt, _ = ctx.attempt(case, key=f"runs:{run.id}:preprocess_image:{i}:{task_id}", task_id=task_id,
                                 attempt_kind="preprocess_image", origin="automatic", provider="kie",
                                 request=None, model_key=model_key, target_kind="run_preprocess",
                                 target_id=run.id, recipe_context={"reference_index": i,
                                                                   "preprocess_mode": run.preprocess_mode})
        if attempt is None:
            continue
        draw = _draw_for(ctx, task_id)
        ctx.submission(attempt, task_id=task_id, call_key=draw.call_key if draw else None)
        if draw is not None:
            ctx.draw_costs(attempt, draw)


def _run_segments(ctx: ImportContext, rs) -> None:
    from ...models import Run

    run = ctx.session.get(Run, rs.run_id)
    case = segment_case(ctx, run, rs)
    status = status_value(rs.status)
    output = None
    if status == "completed" and rs.local_result_path:
        # One output version per provider task: an in-place rerun overwrites
        # result_<i>.mp4 for a new task, which must never be attributed to the
        # previous task's version (and vice versa).
        record = f"import:run_segments:{rs.id}:{rs.seedance_task_id or 'no-task'}"
        existing = store.valid_publication_version(ctx.session, case.id, record, ctx.invalid())
        if existing is not None:
            output = ctx.session.get(KbArtifact, existing.artifact_id)
        else:
            output = ctx.local_artifact(kind="model_output", path=rs.local_result_path, project_id=run.project_id,
                                        what="segment output")
            ctx.append_version(case, output, reason="import", source_record=record)
    if not rs.seedance_task_id:
        if status == "completed" and not rs.source_fallback:
            ctx.report.note("missing_metadata", what="completed segment without a task id", run_segment_id=rs.id)
        return
    model_key, provider_model_id, family = model_provenance(status_value(run.model))
    attempt, _ = ctx.attempt(case, key=f"run_segments:{rs.id}:generation:{rs.seedance_task_id}",
                             task_id=rs.seedance_task_id,
                             attempt_kind="generation", origin="automatic", provider="kie", request=None,
                             model_key=model_key, provider_model_id=provider_model_id, model_family=family,
                             target_kind="run_segment", target_id=rs.id,
                             recipe_context={"segment_def_id": rs.segment_def_id, "index": rs.index,
                                             "request": "unknown (not recorded before capture)"})
    if attempt is None:
        return
    draw = _draw_for(ctx, rs.seedance_task_id)
    ctx.submission(attempt, task_id=rs.seedance_task_id, call_key=draw.call_key if draw else None)
    _terminal_status(ctx, attempt, status, succeeded=("completed",), failed=("failed",), error=rs.error_message,
                     output=output)
    if draw is not None:
        ctx.draw_costs(attempt, draw)
    else:
        ctx.snapshot_cost(attempt, rs.cost_credits, column="run_segments.cost_credits")


def _segment_image_jobs(ctx: ImportContext, job) -> None:
    from ...models import Run, RunSegment

    run, rs = ctx.session.get(Run, job.run_id), ctx.session.get(RunSegment, job.run_segment_id)
    case = segment_case(ctx, run, rs)
    attempt, _ = ctx.attempt(case, key=f"run_segment_image_jobs:{job.id}", task_id=job.kie_task_id,
                             attempt_kind="segment_image",
                             origin="operator", provider="kie", request=None, target_kind="segment_image_job",
                             target_id=job.id, recipe_context={"frame_sec": job.frame_sec,
                                                               "input_count": len(job.input_paths or [])})
    if attempt is None:
        return
    draw = _draw_for(ctx, job.kie_task_id)
    if job.kie_task_id:
        ctx.submission(attempt, task_id=job.kie_task_id, call_key=draw.call_key if draw else None)
    output = None
    if status_value(job.status) == "done" and job.output_path:
        output = ctx.existing_output(attempt) or ctx.local_artifact(
            kind="reference_image", path=job.output_path, project_id=run.project_id, what="segment image output")
    _terminal_status(ctx, attempt, job.status, succeeded=("done",), failed=("failed",), error=job.error_message,
                     output=output)
    if draw is not None:
        ctx.draw_costs(attempt, draw)
    else:
        ctx.snapshot_cost(attempt, job.cost_credits, column="run_segment_image_jobs.cost_credits")


def _range_attempts(ctx: ImportContext, ra) -> None:
    from ...models import Run, RunSegment

    run, rs = ctx.session.get(Run, ra.run_id), ctx.session.get(RunSegment, ra.run_segment_id)
    case = segment_case(ctx, run, rs)
    model_key, provider_model_id, family = model_provenance(status_value(run.model))
    attempt, _ = ctx.attempt(case, key=f"run_segment_range_attempts:{ra.id}", task_id=ra.seedance_task_id,
                             attempt_kind="range_attempt",
                             origin="operator", provider="kie", request=None, model_key=model_key,
                             provider_model_id=provider_model_id, model_family=family, target_kind="range_attempt",
                             target_id=ra.id,
                             recipe_context={"requested_start_sec": ra.requested_start_sec,
                                             "requested_end_sec": ra.requested_end_sec,
                                             "prompt_override_set": ra.prompt_override is not None,
                                             "reference_override_count": len(ra.reference_image_urls_override or [])})
    if attempt is None:
        return
    draw = _draw_for(ctx, ra.seedance_task_id)
    if ra.seedance_task_id:
        ctx.submission(attempt, task_id=ra.seedance_task_id, call_key=draw.call_key if draw else None)
    output = None
    if status_value(ra.status) == "done" and ra.local_result_path:
        output = ctx.existing_output(attempt) or ctx.local_artifact(
            kind="model_output", path=ra.local_result_path, project_id=run.project_id, what="range attempt output")
    _terminal_status(ctx, attempt, ra.status, succeeded=("done",), failed=("failed",), error=ra.error_message,
                     output=output)
    if draw is not None:
        ctx.draw_costs(attempt, draw)
    else:
        ctx.snapshot_cost(attempt, ra.cost_credits, column="run_segment_range_attempts.cost_credits")
    if ra.promoted_at is not None and ra.final_local_path:
        pub = f"range_attempt:{ra.id}"
        if output is not None and not _publication_recorded(ctx, case, pub):
            ctx.append_version(case, output, reason="range_attempt", source_record=pub)
        promotion_version(ctx, run_case(ctx, run), run, path=ra.final_local_path, sha=None,
                          reason="range_attempt", publication_id=pub, published_at=_utc(ra.promoted_at))
        ctx.report.note("missing_metadata", what="range attempt records no base final; no correction event",
                        range_attempt_id=ra.id)


def _audio_patches(ctx: ImportContext, patch) -> None:
    from ...models import Run

    run = ctx.session.get(Run, patch.run_id)
    case = run_case(ctx, run)
    attempt, _ = ctx.attempt(case, key=f"run_audio_patches:{patch.id}", attempt_kind="audio_patch",
                             origin="operator", provider="local", request=None, target_kind="run_audio_patch",
                             target_id=patch.id,
                             recipe_context={"kind": getattr(patch, "kind", None) or "donor",
                                             "filter_params": getattr(patch, "filter_params", None),
                                             "donor_run_id": patch.donor_run_id,
                                             "target_start_sec": patch.target_start_sec,
                                             "target_end_sec": patch.target_end_sec,
                                             "donor_start_sec": patch.donor_start_sec,
                                             "crossfade_sec": patch.crossfade_sec})
    output = None
    if status_value(patch.status) == "done" and patch.output_path:
        output = ctx.existing_output(attempt) or ctx.local_artifact(
            kind="stitched_final", path=patch.output_path, project_id=run.project_id, what="audio patch output",
            duration_sec=patch.result_duration_sec)
    _terminal_status(ctx, attempt, patch.status, succeeded=("done",), failed=("failed",), error=patch.error_message,
                     output=output)
    if patch.promoted_at is not None and patch.output_path:
        pub = f"audio_patch:{patch.id}"
        after, art = promotion_version(ctx, case, run, path=patch.output_path,
                                       sha=output.sha256 if output else None, reason="audio_patch", publication_id=pub,
                                       published_at=_utc(patch.promoted_at))
        _correction(ctx, case, table="run_audio_patches", row_id=patch.id, before_path=patch.base_result_path,
                    before_sha=None, after_artifact=art, after_version=after, method="automatic", publication_id=pub)


def _assistant_promotions(ctx: ImportContext, p) -> None:
    from ...models import Run

    run = ctx.session.get(Run, p.run_id)
    if run is None:
        ctx.report.note("skipped", what="assistant promotion of a deleted run", promotion_id=p.id, run_id=p.run_id)
        return
    case = run_case(ctx, run)
    attempt, _ = ctx.attempt(case, key=f"assistant_run_result_promotions:{p.id}",
                             attempt_kind="run_result_promotion", origin="assistant", provider="local", request=None,
                             target_kind="assistant_promotion", target_id=p.id,
                             recipe_context={"version_id": p.version_id, "delivery": p.delivery, "noop": bool(p.noop)})
    ctx.observe(attempt, kind="status", status="succeeded", payload={"noop": True} if p.noop else None)
    if p.noop:
        return
    before, after = p.before_json or {}, p.after_json or {}
    if not after.get("path"):
        ctx.report.note("missing_metadata", what="assistant promotion without an after path", promotion_id=p.id)
        return
    pub = f"assistant_promotion:{p.id}"
    version, art = promotion_version(ctx, case, run, path=after["path"], sha=_sha(after.get("sha256")),
                                     reason="assistant_promotion", publication_id=pub,
                                     published_at=_utc(p.created_at))
    ctx.observe(attempt, kind="output", output_artifact_id=art.id,
                payload={"publication_id": pub, "case_version_no": version.version_no if version else None})
    _correction(ctx, case, table="assistant_run_result_promotions", row_id=p.id, before_path=before.get("path"),
                before_sha=_sha(before.get("sha256")), after_artifact=art, after_version=version,
                method="assistant", publication_id=pub)


def _derived(ctx: ImportContext, *, table: str, row, run, artifact_kind: str, attempt_kind: str,
             language: Optional[str], snapshot: dict, missing: list) -> None:
    parent = run_case(ctx, run)
    case = ctx.case(case_key=store.derived_case_key(table, row.id), scope="derived", operation=parent.operation,
                    project_id=run.project_id, run_id=run.id, parent_case_id=parent.id, language=language,
                    snapshot={**snapshot, "status_at_import": status_value(row.status), "source": table},
                    missing_facts=["reason_for_edit", *missing])
    path = getattr(row, "output_path", None) or getattr(row, "result_local_path", None)
    output = None
    if status_value(row.status) == "done" and path:
        vs = ctx.versions(case)
        if vs:
            output = ctx.session.get(KbArtifact, vs[0].artifact_id)
        else:
            output = ctx.local_artifact(kind=artifact_kind, path=path, project_id=run.project_id, what=table)
            ctx.append_version(case, output, reason="import", source_record=f"import:{table}:{row.id}")
    record_export(ctx, case, table, row)
    attempt, _ = ctx.attempt(case, key=f"{table}:{row.id}", attempt_kind=attempt_kind, origin="operator",
                             provider="local", request=None, target_kind=table, target_id=row.id)
    _terminal_status(ctx, attempt, row.status, succeeded=("done",), failed=("failed",),
                     error=getattr(row, "error_message", None) or getattr(row, "error", None), output=output)


def _tail_edits(ctx: ImportContext, te) -> None:
    from ...models import Run

    _derived(ctx, table="run_tail_edits", row=te, run=ctx.session.get(Run, te.run_id), artifact_kind="tail_edit",
             attempt_kind="tail_edit", language=None,
             snapshot={"cut_sec": te.cut_sec, "library_clip_id": te.library_clip_id,
                       "source_type": status_value(te.source_type), "source_caption_id": te.source_caption_id,
                       "gdrive_file_id": te.gdrive_file_id}, missing=[])


def _captions(ctx: ImportContext, cap) -> None:
    from ...models import Run

    _derived(ctx, table="run_captions", row=cap, run=ctx.session.get(Run, cap.run_id), artifact_kind="caption",
             attempt_kind="caption", language=cap.language,
             snapshot={"lines_source": cap.lines_source, "lines_reused_from": cap.lines_reused_from,
                       "source_type": status_value(cap.source_type), "source_tail_edit_id": cap.source_tail_edit_id,
                       "style_version": cap.style_version, "line_count": len(cap.lines or []),
                       "gdrive_file_id": cap.gdrive_file_id}, missing=[])


def overlay_run(ctx: ImportContext, job):
    from ...models import Run, RunCaption, RunTailEdit

    run_id = job.parent_run_id or job.source_run_id
    if run_id is None and job.source_tail_edit_id:
        te = ctx.session.get(RunTailEdit, job.source_tail_edit_id)
        run_id = te.run_id if te else None
    if run_id is None and job.source_caption_id:
        cap = ctx.session.get(RunCaption, job.source_caption_id)
        run_id = cap.run_id if cap else None
    return ctx.session.get(Run, run_id) if run_id else None


def _overlays(ctx: ImportContext, job) -> None:
    run = overlay_run(ctx, job)
    if run is None:
        ctx.report.note("skipped", what="overlay with no resolvable run (no project to scope it)", overlay_id=job.id)
        return
    missing = ["output_media_deleted"] if job.deleted_at is not None else []
    _derived(ctx, table="overlay_jobs", row=job, run=run, artifact_kind="overlay", attempt_kind="overlay",
             language=run.language,
             snapshot={"source_type": status_value(job.source_type), "overlay_count": len(job.overlays or []),
                       "deleted": job.deleted_at is not None, "gdrive_file_id": job.gdrive_file_id}, missing=missing)


def manual_provenance(ctx: ImportContext, mc):
    """(run, project_id, via): the row's own copied provenance first, the overlay job second.

    ``manual_creatives`` has no run/project columns; registration copies a
    snapshot into ``details`` so tracing survives overlay deletion. Only ids a
    script put there (``run_id``/``parent_run_id``/``source_run_id``,
    ``project_id``) are used; nothing is guessed from names.
    """
    from ...models import OverlayJob, Run

    details = mc.details if isinstance(mc.details, dict) else {}
    for key in ("run_id", "parent_run_id", "source_run_id"):
        run = ctx.session.get(Run, details[key]) if isinstance(details.get(key), str) else None
        if run is not None:
            return run, run.project_id, f"details.{key}"
    if isinstance(details.get("project_id"), str) and details["project_id"]:
        return None, details["project_id"], "details.project_id"
    job = ctx.session.get(OverlayJob, mc.overlay_job_id) if mc.overlay_job_id else None
    run = overlay_run(ctx, job) if job is not None else None
    return run, (run.project_id if run is not None else None), "overlay_job"


def _manual_creatives(ctx: ImportContext, mc) -> None:
    run, project_id, via = manual_provenance(ctx, mc)
    if project_id is None:
        ctx.report.note("skipped", what="manual creative with no resolvable project", manual_creative_id=mc.id,
                        linear_issue=mc.linear_issue)
        return
    from ...models import VideoProject

    parent = run_case(ctx, run) if run is not None else None
    operation = parent.operation if parent is not None else operation_for(ctx.session.get(VideoProject, project_id))
    case = ctx.case(case_key=store.manual_case_key(mc.id), scope="manual", operation=operation, project_id=project_id,
                    run_id=run.id if run is not None else None,
                    parent_case_id=parent.id if parent is not None else None, language=mc.language,
                    snapshot={"pipeline_type": mc.pipeline_type, "idea": mc.idea, "linear_issue": mc.linear_issue,
                              "overlay_job_id": mc.overlay_job_id, "overlay_asset_name": mc.overlay_asset_name,
                              "source_link": mc.source_link, "attributed_via": via, "source": "manual_creatives"},
                    missing_facts=["script_recipe", "reason_for_edit"] + ([] if run is not None else ["run"]))
    vs = ctx.versions(case)
    if vs:
        output = ctx.session.get(KbArtifact, vs[0].artifact_id)
    else:
        output = ctx.artifact(kind="delivered_final", locator_kind="drive_file", locator=mc.gdrive_file_id,
                              project_id=project_id)
        ctx.append_version(case, output, reason="import", source_record=f"import:manual_creatives:{mc.id}")
    record_export(ctx, case, "manual_creatives", mc)
    attempt, _ = ctx.attempt(case, key=f"manual_creatives:{mc.id}", attempt_kind="manual_script", origin="operator",
                             provider="local", request=None, target_kind="manual_creative", target_id=mc.id)
    ctx.observe(attempt, kind="output", output_artifact_id=output.id)


DRAW_ATTEMPT_KINDS = {"video": "generation", "image": "preprocess_image", "range video": "range_attempt",
                      "segment image": "segment_image"}


def draw_case(ctx: ImportContext, draw):
    """The case a paid call belongs to, from its reservation scope; None = unattributable."""
    from ...models import KieSpendReservation, Run, RunCaption, RunSegment, RunSegmentImageJob, RunSegmentRangeAttempt

    res = ctx.session.get(KieSpendReservation, draw.reservation_id)
    scope_key = res.scope_key if res else ""
    prefix, _, ref = scope_key.partition(":")
    run = None
    if prefix == "range":
        ra = ctx.session.get(RunSegmentRangeAttempt, ref)
        if ra is not None:
            return segment_case(ctx, ctx.session.get(Run, ra.run_id), ctx.session.get(RunSegment, ra.run_segment_id)), \
                ctx.session.get(Run, ra.run_id)
    elif prefix == "segimg":
        job = ctx.session.get(RunSegmentImageJob, ref)
        if job is not None:
            return segment_case(ctx, ctx.session.get(Run, job.run_id), ctx.session.get(RunSegment, job.run_segment_id)), \
                ctx.session.get(Run, job.run_id)
    elif prefix == "caption":
        cap = ctx.session.get(RunCaption, ref)
        case = store.current_case(ctx.session, store.derived_case_key("run_captions", ref))
        if case is not None:
            return case, ctx.session.get(Run, cap.run_id) if cap else None
    run_id = (res.run_id if res else None) or (ref if prefix == "run" else None)
    if run_id:
        run = ctx.session.get(Run, run_id)
        if run is not None:
            return run_case(ctx, run), run
        case = store.current_case(ctx.session, store.run_case_key(run_id))
        if case is not None:
            return case, None
    return None, None


def _unattributable(ctx: ImportContext, draw) -> None:
    """Counted apart, durably in the batch report, so coverage (pilot gate (a))
    sees every unattributable paid draw and its credits. The cursor tally
    (committed with this row's cursor advance) is the durable record; keyed by
    draw id, so a revisit never counts it twice."""
    ctx.cursor_tally[f"unattributable:{draw.id}"] = int(draw.mcredits)
    stats = ctx.report.extra.setdefault("reconcile", {})
    ctx.report.bump(stats, "unattributable_draws")
    ctx.report.bump(stats, "unattributable_draws_mcredits", int(draw.mcredits))
    ctx.report.note("skipped", what="paid call with no attributable case", draw_id=draw.id, draw_kind=draw.kind,
                    mcredits=draw.mcredits)


def _spend_draws(ctx: ImportContext, draw) -> None:
    """Coverage (§6.1): a paid generation call without an attempt becomes an ``ambiguous`` one.

    Matching is by exact call key (= kie task id) only, so a draw whose task id
    never reached its row (worker killed after acceptance) stays ambiguous and
    is never attributed to a later attempt of the same segment.
    """
    attempt = ctx.attempt_for_call_key(draw.call_key)
    if attempt is not None:
        ctx.draw_costs(attempt, draw)
        return
    # A second provider's paid call (Fal, AICP-330: kind "fal_<family>", scope
    # "fal:<task id>") is a GENERATION, not an excluded LLM call; until its
    # consumer links tasks to cases it is counted as unattributable, so coverage
    # sees it and its credits.
    if (getattr(draw, "provider", None) or "kie") != "kie":
        _unattributable(ctx, draw)
        return
    if draw.kind not in DRAW_ATTEMPT_KINDS:
        # LLM calls (transcribe/translate/captions) carry random call keys and are
        # never captured as attempts; counted apart so coverage is not failed by construction.
        ctx.report.bump(ctx.report.extra.setdefault("reconcile", {}), "llm_draws_excluded")
        ctx.report.bump(ctx.report.extra["reconcile"], "llm_draws_excluded_mcredits", int(draw.mcredits))
        return
    case, run = draw_case(ctx, draw)
    if case is None:
        # Its Run is gone (Runs are hard-deleted) or its scope names nothing: no case
        # can ever hold it.
        _unattributable(ctx, draw)
        return
    kind = DRAW_ATTEMPT_KINDS[draw.kind]
    model = model_provenance(status_value(run.model)) if run is not None and kind in ("generation", "range_attempt") \
        else (None, None, None)
    attempt, _ = ctx.attempt(case, key=f"kie_spend_draws:{draw.id}:ambiguous", attempt_kind=kind,
                             origin="automatic", provider="kie", request=None, model_key=model[0],
                             provider_model_id=model[1], model_family=model[2], target_kind="spend_draw",
                             target_id=draw.id, recipe_context={"draw_kind": draw.kind})
    ctx.observe(attempt, kind="ambiguous", payload={"reason": "paid call with no recorded attempt; request unknown",
                                                    "call_key": draw.call_key, "draw_kind": draw.kind})
    ctx.submission(attempt, task_id=draw.call_key, call_key=draw.call_key)
    ctx.draw_costs(attempt, draw)
    ctx.report.bump(ctx.report.extra.setdefault("reconcile", {}), "ambiguous_draw_attempts")


def _models():
    from ...assistant_models import AssistantRunResultPromotion
    from ...models import (
        KieSpendDraw, ManualCreative, OverlayJob, Run, RunAudioPatch, RunCaption, RunSegment, RunSegmentImageJob,
        RunSegmentRangeAttempt, RunTailEdit,
    )
    return {"runs": Run, "run_segments": RunSegment, "segment_image_jobs": RunSegmentImageJob,
            "range_attempts": RunSegmentRangeAttempt, "audio_patches": RunAudioPatch,
            "assistant_promotions": AssistantRunResultPromotion, "captions": RunCaption, "tail_edits": RunTailEdit,
            "overlays": OverlayJob, "manual_creatives": ManualCreative, "spend_draws": KieSpendDraw}


def _ready(terminal: tuple):
    """False = imported as it stands now and revisited until it is terminal."""
    return lambda row, now: _terminal(getattr(row, "status", None), terminal)


def _always(row, now) -> bool:
    return True


def _assistant_media(ctx, p) -> list:
    after = p.after_json or {}
    return [(after.get("path"), _sha(after.get("sha256")) is None)] if not p.noop else []


def _overlay_media(ctx, job) -> list:
    return [(job.result_local_path, True)]


SOURCES = (
    # failed/incomplete runs can still be retried to done, so only done ends the revisit
    Source("runs", lambda: _models()["runs"], _ready(("done",)), _runs, True, ages_out=True, media=_runs_media, cursor_col="updated_at"),
    Source("run_segments", lambda: _models()["run_segments"], _ready(("completed", "failed", "skipped")),
           _run_segments, True, media=lambda ctx, rs: [(rs.local_result_path, True)], cursor_col="updated_at"),
    Source("segment_image_jobs", lambda: _models()["segment_image_jobs"], _ready(("done", "failed")),
           _segment_image_jobs, True, media=lambda ctx, j: [(j.output_path, True)], cursor_col="updated_at"),
    Source("range_attempts", lambda: _models()["range_attempts"], _ready(("done", "failed")), _range_attempts, True,
           media=lambda ctx, ra: [(ra.local_result_path, True), (ra.final_local_path, True)], cursor_col="updated_at"),
    Source("audio_patches", lambda: _models()["audio_patches"], _ready(("done", "failed")), _audio_patches,
           media=lambda ctx, p: [(p.output_path, True)], cursor_col="updated_at"),
    Source("assistant_promotions", lambda: _models()["assistant_promotions"], _always, _assistant_promotions,
           media=_assistant_media),
    Source("captions", lambda: _models()["captions"], _ready(("done", "failed")), _captions,
           media=lambda ctx, c: [(c.output_path, True)]),
    Source("tail_edits", lambda: _models()["tail_edits"], _ready(("done", "failed")), _tail_edits,
           media=lambda ctx, t: [(t.output_path, True)]),
    Source("overlays", lambda: _models()["overlays"], _ready(("done", "failed")), _overlays, media=_overlay_media),
    Source("manual_creatives", lambda: _models()["manual_creatives"], _always, _manual_creatives),
    Source("spend_draws", lambda: _models()["spend_draws"], _always, _spend_draws),
)
SOURCE_NAMES = tuple(s.name for s in SOURCES)
