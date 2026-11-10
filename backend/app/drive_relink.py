"""Re-link a run to a NEW Drive file when its delivered file was deleted (AICP-348).

Since AICP-342 every re-delivery of a run (segment rerun, take restore, range
attempt, audio patch, retry) replaces the CONTENT behind the run's existing
Drive id and never falls back to a fresh upload, because a fresh upload would
silently change the Creative Link an exported sheet row holds. When that Drive
file has been deleted, the update returns 404 and the delivery fails on purpose.
This module is the explicit, audited way out:

1. ``GET /api/v2/runs/{rid}/drive-file`` — :func:`inspect`: a FRESH Drive
   metadata read (``present`` / ``trashed`` / ``missing``), the exports the DB
   knows of that referenced the run's link, and the re-link history. Advisory.
2. ``POST /api/v2/runs/{rid}/drive-relink`` — admission in ``api_v2``: under
   the Run-row lock (``_lock_run_for_result_mutation``) it applies
   :func:`blocking_reason`, then commits a ``queued``
   :class:`~app.models.RunDriveRelink` row BEFORE any Drive call and enqueues
   :func:`process_relink`. A ``queued`` row whose job is gone (worker killed)
   is resumed by the next POST rather than duplicated.
3. :func:`process_relink` (RQ, ``overlay`` queue) — the AICP-342 in-place
   lock order: Run-row lock (``with_for_update``), then every check again
   (status, the Drive id the operator confirmed, the final, no active range
   attempt / audio patch / segment fill / assistant publication), then the
   fresh 404 check, then the Drive side effect, then one commit that re-points
   ``Run.result_gdrive_file_id`` and marks the row ``done``.

The fence is the Run-row lock plus the status check, not the run's Redis
processing lock: every admission that can start a delivery (retry, rerun,
restore, range attempt, audio patch, assistant publication) takes the same
Run-row lock and moves the run out of ``done``/``failed`` or creates an
active row this job refuses on, and finalize's own in-place update refuses
unless the run is ``delivering`` under that lock. Taking the Redis lock as
well would add a hand-off race (a retry admitted between this commit and the
unlock would have its submit job skip on the held lock) and nothing else.

**Idempotent create.** Admission reserves the new file's id with Drive
``generateIds`` and stores it (``target_file_id``); the worker CREATES the file
with exactly that id. A second create with the same id cannot make a second
file, and every "did it land?" question — a lost response, a retry, a commit
failure after the upload, a killed worker resumed by the next POST — is a
strongly consistent ``files.get`` by that id (not an eventually consistent
search). A file found there is adopted (its content fixed by md5 if the final
changed). A failed upload never ends the row: it stays ``queued`` with the
error (a request Drive is still processing could still create the reserved
file), and the next press resumes it with the same id. A row that does end
unlinked (``refused``) hands its reserved id on to the next re-link of the
same old file, so even an upload of it that lands after the refusal is
adopted rather than duplicated.

**404 is not proof of deletion.** Drive also answers 404 for a file the
service account can no longer read. :func:`confirm_missing` therefore also
requires the run's Drive folder to be readable: an unreadable folder means an
access problem (e.g. a rotated service account), where a re-link would replace
a Creative Link that still works for everyone else. A file moved out of the
folder or unshared individually still reads as deleted; the confirmation text
tells the operator so. The upload also carries the row
id as an ``appProperties`` tag, for traceability only.

**Status is not changed.** A ``failed`` run stays failed: ``failed`` does not
prove the only failure was the delivery, and marking it ``done`` would skip
finalize's completeness/freshness gates. The operator then presses Retry Run,
which resumes the pipeline (usually a delivery-only finalize that replaces the
content behind the NEW id — it may do more if segments need it).

**Sheet rows are listed, never edited** (:func:`scan_sheet`): a point-in-time
read of the Final Creatives tab for cells holding the old file id, plus the
exports the DB knows of. Editing cells in a sheet that people and the
downstream renamer (ai-fb-audit) also edit, keyed by a link that may appear in
several rows, is left to a human. An export job already enqueued before the
re-link reads the id when it runs, so it appends the NEW link; one that read
the old id just before the commit could still append the old one after the
scan — rescan (GET) before fixing rows.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import languages, overlay_runs
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient, GDriveError
from .models import (
    OverlayJob,
    Run,
    RunAudioPatch,
    RunCaption,
    RunDriveRelink,
    RunSegmentRangeAttempt,
    RunTailEdit,
)
from .state_machine import AUDIO_PATCH_ACTIVE_STATUSES, RANGE_ATTEMPT_ACTIVE_STATUSES

log = logging.getLogger(__name__)

APP_PROPERTY = "reskinRelinkId"
RELINKABLE_RUN_STATUSES = ("done", "failed")
UPLOAD_ATTEMPTS = 3
UPLOAD_BACKOFF_SEC = 5.0

ACTIVE = "queued"
TERMINAL = ("done", "refused")

# Pipeline Type cells, as the export jobs write them (app/tasks.py).
_PT_RUN = "re-skin-v2"
_PT_TAIL = "re-skin-tail-edit"
_PT_CAPTION = "re-skin-captions"
_PT_OVERLAY = "re-skin-overlay"


def drive_link(file_id: Optional[str]) -> Optional[str]:
    return f"https://drive.google.com/file/d/{file_id}/view" if file_id else None


def _status_text(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


# ---------------------------------------------------------------------------
# What may block a re-link — shared by the API admission and the worker
# ---------------------------------------------------------------------------


def blocking_reason(db: Session, run: Run) -> Optional[str]:
    """Why *run* cannot be re-linked right now, or None. Call it under the
    Run-row lock (the admissions it guards against take the same lock)."""
    status = _status_text(run.status)
    if status not in RELINKABLE_RUN_STATUSES:
        return (
            f"run status is {status!r}; a re-link needs a 'done' or 'failed' run "
            "(nothing may be delivering it meanwhile)"
        )
    if not run.result_gdrive_file_id:
        return "the run has no Drive file to re-link (it was never delivered)"
    if not run.result_local_path or not os.path.isfile(run.result_local_path):
        return (
            "the run's final video is not on disk, so there is nothing to upload "
            "— retry the run to rebuild it"
        )
    active_attempt = db.execute(
        select(RunSegmentRangeAttempt.id).where(
            RunSegmentRangeAttempt.run_id == run.id,
            RunSegmentRangeAttempt.status.in_([s.value for s in RANGE_ATTEMPT_ACTIVE_STATUSES]),
        ).limit(1)
    ).scalar_one_or_none()
    if active_attempt is not None:
        return f"a range regeneration ({active_attempt}) is in progress on this run"
    active_patch = db.execute(
        select(RunAudioPatch.id).where(
            RunAudioPatch.run_id == run.id,
            RunAudioPatch.status.in_([s.value for s in AUDIO_PATCH_ACTIVE_STATUSES]),
        ).limit(1)
    ).scalar_one_or_none()
    if active_patch is not None:
        return f"an audio patch ({active_patch}) is in progress on this run"
    from .pipeline_segment_fill import active_fill

    fill = active_fill(db, run.id)
    if fill is not None:
        return f"segment caption fill {fill.id} is {fill.status} on this run"
    from .assistant_run_result import ensure_no_unresolved_intent
    from .assistant_store import AssistantConflict

    try:
        ensure_no_unresolved_intent(db, run.id)
    except AssistantConflict as exc:
        return str(exc)
    return None


def active_relink(db: Session, run_id: str) -> Optional[RunDriveRelink]:
    return db.execute(
        select(RunDriveRelink).where(
            RunDriveRelink.run_id == run_id, RunDriveRelink.status == ACTIVE,
        ).limit(1)
    ).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Rows that carry the run's link
# ---------------------------------------------------------------------------


def expected_export_rows(db: Session, run: Run) -> list[dict]:
    """Exports the DB knows of whose row referenced the run's Drive link.

    The run's own row has it as ``Creative Link``; exported tail edits,
    captions and overlays made directly from the run carry it as ``Input
    Link`` (an overlay made from a tail edit or a caption carries THAT file's
    link instead, so it is not listed). A stamp says an export was admitted,
    not what the sheet holds now — :func:`scan_sheet` is the ground truth.
    """
    rows: list[dict] = []
    if run.final_creatives_exported_at is not None:
        rows.append({"kind": "original", "id": run.id, "job_id": run.short_id,
                     "column": "Creative Link", "pipeline_type": _PT_RUN})
    for edit_id in db.execute(
        select(RunTailEdit.id).where(
            RunTailEdit.run_id == run.id, RunTailEdit.final_creatives_exported_at.isnot(None),
        ).order_by(RunTailEdit.created_at)
    ).scalars():
        rows.append({"kind": "tail", "id": edit_id,
                     "job_id": languages.job_id_for(edit_id, run.language),
                     "column": "Input Link", "pipeline_type": _PT_TAIL})
    for caption_id in db.execute(
        select(RunCaption.id).where(
            RunCaption.run_id == run.id, RunCaption.final_creatives_exported_at.isnot(None),
        ).order_by(RunCaption.created_at)
    ).scalars():
        rows.append({"kind": "captioned", "id": caption_id,
                     "job_id": languages.job_id_for(caption_id, run.language),
                     "column": "Input Link", "pipeline_type": _PT_CAPTION})
    overlays = list(db.execute(
        select(OverlayJob).where(
            OverlayJob.source_run_id == run.id,
            OverlayJob.final_creatives_exported_at.isnot(None),
        ).order_by(OverlayJob.created_at)
    ).scalars())
    if overlays:
        siblings = overlay_runs.count_overlay_jobs_for_run(run.id, db)
        for job in overlays:
            rows.append({"kind": "overlay", "id": job.id,
                         "job_id": overlay_runs.sheet_job_id(job, run, sibling_count=siblings),
                         "column": "Input Link", "pipeline_type": _PT_OVERLAY})
    return rows


def scan_sheet(file_id: str, *, client=None) -> dict:
    """Point-in-time scan of the Final Creatives tab for cells holding
    *file_id*. Read-only. ``{"scanned_at", "rows": [{"row", "column",
    "job_id", "pipeline_type", "value"}], "error"}`` — ``row`` is the 1-based
    sheet row. A read failure is reported in ``error``, never raised."""
    out: dict = {"scanned_at": datetime.now(timezone.utc).isoformat(), "rows": [], "error": None}
    sid = settings.FINAL_CREATIVES_SPREADSHEET_ID
    tab = settings.FINAL_CREATIVES_TAB
    if not sid:
        out["error"] = "FINAL_CREATIVES_SPREADSHEET_ID is not configured"
        return out
    try:
        if client is None:
            from .sheets_client import SheetsClient

            client = SheetsClient()
        header = client.get_header(sid, tab)
        norm = {str(h).strip().lower(): i for i, h in enumerate(header or [])}

        def column(name: str) -> list[str]:
            idx = norm.get(name)
            return client.column_values(sid, tab, idx) if idx is not None else []

        job_ids = column("job id")
        types = column("pipeline type")
        for name in ("Creative Link", "Input Link"):
            for i, value in enumerate(column(name.lower())):
                if i == 0 or file_id not in (value or ""):
                    continue  # i == 0 is the header row
                out["rows"].append({
                    "row": i + 1, "column": name,
                    "job_id": job_ids[i] if i < len(job_ids) else "",
                    "pipeline_type": types[i] if i < len(types) else "",
                    "value": value,
                })
    except Exception as exc:  # noqa: BLE001 - advisory read
        out["error"] = f"could not read the Final Creatives sheet: {exc}"
    out["rows"].sort(key=lambda r: (r["row"], r["column"]))
    return out


def confirm_missing(gdrive, file_id: str, folder_id: Optional[str]) -> Optional[str]:
    """None when Drive reports *file_id* missing AND the run's Drive folder is
    still readable, else why not. Raises nothing.

    Drive answers 404 both for a deleted file and for one the service account
    can no longer read. A readable folder makes "deleted (or moved/unshared
    individually)" the explanation; an unreadable folder too means an ACCESS
    problem (e.g. a rotated service account), where a re-link would replace a
    Creative Link that still works for everyone else.
    """
    try:
        state = gdrive.file_state(file_id)
    except Exception as exc:  # noqa: BLE001 - never guess "missing"
        return f"could not confirm the Drive file is missing ({exc})"
    if state != "missing":
        return (f"the Drive file is {state}, not missing — a re-link is only for a "
                "file Drive reports deleted"
                + (" (restore it from the Drive trash instead)" if state == "trashed" else ""))
    if not folder_id:
        return "no Drive folder is configured for this run"
    try:
        folder_state = gdrive.file_state(folder_id)
    except Exception as exc:  # noqa: BLE001
        return f"could not check the run's Drive folder ({exc})"
    if folder_state != "present":
        return (f"the run's Drive folder {folder_id} is {folder_state} for the service "
                "account too — this looks like an access problem (e.g. a changed service "
                "account), not a deleted file; fix the sharing instead")
    return None


# ---------------------------------------------------------------------------
# GET — fresh check
# ---------------------------------------------------------------------------


def relink_view(row: RunDriveRelink) -> dict:
    return {
        "id": row.id,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "old_file_id": row.old_file_id,
        "old_link": drive_link(row.old_file_id),
        "new_file_id": row.new_file_id,
        "new_link": drive_link(row.new_file_id),
        "run_status": row.run_status,
        "error_message": row.error_message,
        "exported_rows": row.exported_rows or [],
        "sheet_scan": row.sheet_scan,
    }


def inspect(db: Session, run: Run, *, gdrive=None, sheets=None, scan: bool = True) -> dict:
    """Fresh state of the run's Drive file, what a re-link would touch, and the
    history. Advisory: the POST and the worker re-check everything."""
    file_id = run.result_gdrive_file_id
    state, error, reason = None, None, None
    gdrive = gdrive or GDriveClient()
    if file_id:
        try:
            state = gdrive.file_state(file_id)
        except Exception as exc:  # noqa: BLE001 - reported, never guessed
            state, error = "unknown", f"Drive did not answer: {exc}"
        reason = confirm_missing(
            gdrive, file_id, run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID,
        )
    else:
        reason = "the run has no Drive file"
    if reason is None:
        reason = blocking_reason(db, run)
    current = active_relink(db, run.id)
    resumable = False
    if reason is None and current is not None:
        # A queued row whose job died (deploy, enqueue failure) is resumed by
        # the POST; only a live one blocks.
        from .tasks import drive_relink_job_live

        try:
            live = drive_relink_job_live(current.id)
        except Exception as exc:  # noqa: BLE001 - never guess "dead"
            reason = f"could not check the queued re-link {current.id}: {exc}"
        else:
            if live:
                reason = f"a re-link ({current.id}) is in progress"
            else:
                resumable = True
    history = list(db.execute(
        select(RunDriveRelink).where(RunDriveRelink.run_id == run.id)
        .order_by(RunDriveRelink.created_at.desc())
    ).scalars())
    return {
        "run_id": run.id,
        "job_id": run.short_id,
        "run_status": _status_text(run.status),
        "file_id": file_id,
        "link": drive_link(file_id),
        "state": state,
        "error": error,
        "can_relink": reason is None,
        "resumes": current.id if resumable else None,
        "reason": reason,
        "exported_rows": expected_export_rows(db, run),
        "sheet_scan": scan_sheet(file_id, client=sheets) if (file_id and scan) else None,
        "relinks": [relink_view(r) for r in history],
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _end(db: Session, row: RunDriveRelink, status: str, message: str) -> str:
    row.status = status
    row.error_message = message
    db.commit()
    log.warning("drive relink %s (run %s): %s — %s", row.id, row.run_id, status, message)
    return status


def _adopt(gdrive, file_id: str, pin: str, local_md5: str) -> str:
    """The file with *file_id* exists: make sure it holds *pin*'s bytes."""
    try:
        remote = gdrive.file_md5(file_id)
    except Exception as exc:  # noqa: BLE001 - fix the content when unsure
        log.warning("drive relink: md5 of %s unreadable (%s) — re-sending", file_id, exc)
        remote = None
    if remote != local_md5:
        gdrive.update_file_content(file_id, pin)
    return file_id


def _upload_or_adopt(gdrive, row: RunDriveRelink, pin: str, folder_id: str,
                     sleep=time.sleep) -> str:
    """Create THIS row's Drive file (id ``row.target_file_id``) at most once.

    Before every attempt, ``files.get`` the target id: if it exists, an earlier
    attempt of this same row landed (its response or its commit was lost) —
    adopt it. A check Drive does not answer raises :class:`GDriveError`
    (unknown, never "absent")."""
    target = row.target_file_id
    local_md5 = _md5_of(pin)
    last_exc: Optional[Exception] = None
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        state = gdrive.file_state(target)
        if state == "trashed":
            raise GDriveError(f"the new Drive file {target} is in the trash")
        if state == "present":
            return _adopt(gdrive, target, pin, local_md5)
        try:
            result = gdrive.upload_file(
                pin, folder_id, f"reskin_run_{row.run_id}.mp4",
                app_properties={APP_PROPERTY: row.id}, file_id=target,
            )
            got = (result or {}).get("id")
            if got != target:
                raise GDriveError(f"Drive created {got!r}, not the reserved id {target!r}")
            return target
        except Exception as exc:  # noqa: BLE001 - settled by files.get next round
            last_exc = exc
            log.warning("drive relink %s: upload attempt %d/%d failed: %s",
                        row.id, attempt, UPLOAD_ATTEMPTS, exc)
            if attempt < UPLOAD_ATTEMPTS:
                sleep(UPLOAD_BACKOFF_SEC * attempt)
    # The last attempt may have landed with its response lost.
    if gdrive.file_state(target) == "present":
        return _adopt(gdrive, target, pin, local_md5)
    raise last_exc  # type: ignore[misc]


def process_relink(relink_id: str, *, gdrive=None, sheets=None, sleep=time.sleep) -> str:
    """Carry out one queued re-link. Returns the row's resulting status."""
    # AICP-350: a running re-link has no status the deploy drain could count,
    # so it holds the admission gate's share lock for its whole run (a deploy's
    # begin.sql waits for it) and does not start while the gate is paused.
    from . import ops_gate

    with ops_gate.job_barrier(get_session) as admitted:
        if not admitted:
            from .tasks import DRIVE_RELINK_JOB_TIMEOUT, QUEUE_OVERLAY, defer_uncounted_job

            defer_uncounted_job(QUEUE_OVERLAY, "app.tasks.run_drive_relink", relink_id,
                                job_timeout=DRIVE_RELINK_JOB_TIMEOUT)
            return "deferred"
        return _process_relink(relink_id, gdrive=gdrive, sheets=sheets, sleep=sleep)


def _process_relink(relink_id: str, *, gdrive=None, sheets=None, sleep=time.sleep) -> str:
    with get_session() as db:
        row = db.get(RunDriveRelink, relink_id)
        if row is None:
            raise ValueError(f"RunDriveRelink not found: {relink_id}")
        if row.status != ACTIVE:
            return row.status
        run = db.get(Run, row.run_id)
        if run is None:
            return _end(db, row, "refused", "the run was deleted")

        # AICP-342 lock order: the Run row first, then re-check everything.
        db.refresh(run, with_for_update=True)
        db.refresh(row, with_for_update=True)
        if row.status != ACTIVE:
            db.rollback()
            return row.status
        if gdrive is None:
            gdrive = GDriveClient()

        def refuse(message: str) -> str:
            # A RESUMED row may already have created its reserved file: say so,
            # never leave an unlinked copy unmentioned.
            try:
                made = gdrive.file_state(row.target_file_id) == "present"
            except Exception:  # noqa: BLE001
                made = None
            if made:
                message += (f"; an earlier attempt already created Drive file "
                            f"{row.target_file_id}, which is NOT linked — delete it if unwanted")
            elif made is None:
                message += (f"; could not check whether Drive file {row.target_file_id} "
                            "was created by an earlier attempt")
            else:
                message += " — nothing uploaded"
            return _end(db, row, "refused", message)

        if run.result_gdrive_file_id != row.old_file_id:
            return refuse("the run's Drive file changed since the re-link was confirmed "
                          f"(now {run.result_gdrive_file_id!r})")
        if run.result_local_path != row.result_path:
            return refuse("the run's final changed since the re-link was confirmed — "
                          "check the run and confirm again")
        reason = blocking_reason(db, run)
        if reason is not None:
            return refuse(reason)
        folder_id = run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
        reason = confirm_missing(gdrive, row.old_file_id, folder_id)
        if reason is not None:
            return refuse(reason)

        from .pipeline_v2 import _sweep_stale_delivery_pins

        pin_dir = os.path.dirname(row.result_path)
        _sweep_stale_delivery_pins(pin_dir)
        pin = os.path.join(pin_dir, f".deliver-{uuid.uuid4().hex}.mp4")
        try:
            os.link(row.result_path, pin)
            try:
                new_id = _upload_or_adopt(gdrive, row, pin, folder_id, sleep=sleep)
            except Exception as exc:  # noqa: BLE001
                try:
                    landed = gdrive.file_state(row.target_file_id)
                except Exception:  # noqa: BLE001
                    landed = None
                # Never terminal once an upload was attempted: a request Drive
                # is still processing could create the reserved file later, so
                # the row keeps its id and the next press resumes THIS row
                # (adopting the file if it landed) instead of reserving a
                # second id.
                row.error_message = (
                    f"upload failed ({exc}); Drive file {row.target_file_id} "
                    + ("does not exist yet" if landed == "missing" else "may exist")
                    + " — press the re-link again to resume it"
                )
                db.commit()
                log.error("drive relink %s: %s", row.id, row.error_message)
                return ACTIVE
        finally:
            try:
                os.remove(pin)
            except OSError:
                pass

        old_id = run.result_gdrive_file_id
        run.result_gdrive_file_id = new_id
        row.new_file_id = new_id
        row.exported_rows = expected_export_rows(db, run)
        row.status = "done"
        row.error_message = None
        db.commit()
        log.warning(
            "drive relink %s: run %s re-linked from deleted Drive file %s to NEW file %s "
            "(Creative Link changed; Job ID %s unchanged)",
            row.id, run.id, old_id, new_id, run.short_id,
        )

        # After the commit and outside the lock: where the old link still is.
        row.sheet_scan = scan_sheet(old_id, client=sheets)
        try:
            db.commit()
        except Exception as exc:  # noqa: BLE001 - the re-link itself is committed
            db.rollback()
            log.warning("drive relink %s: could not store the sheet scan: %s", row.id, exc)
        return "done"
