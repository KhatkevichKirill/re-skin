"""
pipeline_audio_patch.py — process_audio_patch(patch_id) for the "Fix audio from
another run" tool.

Seedance writes a localisation run's speech itself, and it occasionally
mispronounces a brand name in an otherwise good take while a sibling run of the
same project — same script, same driving footage — says it correctly. This
module splices that one window of the sibling's audio into the run's delivered
final and makes the result the run's OWN result: the video stream is copied
untouched, no audio sample outside the window moves, and the Drive file's
CONTENT is replaced so the same link (and any Final Creatives row already
holding it) now serves the fixed video.

What it deliberately does NOT do
--------------------------------
Produce a derived creative. There is no new Job ID, no sheet row and no new
Drive file, because a take with the brand mispronounced is a defect and not a
deliverable worth tracking beside the good one — see docs/run-audio-patch.md
§ "Why this is not a derived creative". Everything already built on the run
(the export button, Sent-to-Final, tail edits, captions, overlays, automation
copies) therefore keeps working with no changes at all.

Shape
-----
Mirrors pipeline_tail.process_tail_edit for the job itself (a single ffmpeg
pass, a Drive call, ``queued -> processing -> delivering -> done``, orphan
resume, bounded error messages) and pipeline_range_attempt for the ENDING: the
candidate is written under ``storage.audio_patch_dir`` and only
:func:`_promote` — under a Run row lock, after both the render and the Drive
update have succeeded — re-points the Run at it. A failed patch leaves the
run's delivered result byte-for-byte as it was.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from . import media as media_mod
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .knowledge.publication import PublishedFinal
from .models import Run, RunAudioPatch
from .state_machine import AudioPatchStatus, RunStatus, transition
from .storage import audio_patch_dir

log = logging.getLogger(__name__)

# Delivery retry tuning — mirrors pipeline_tail's TAIL_EDIT_DELIVER_* so a
# transient Drive error self-heals the same way a tail edit's delivery does.
AUDIO_PATCH_DELIVER_ATTEMPTS = int(os.getenv("AUDIO_PATCH_DELIVER_ATTEMPTS", "3"))
AUDIO_PATCH_DELIVER_BACKOFF_SEC = float(
    os.getenv("AUDIO_PATCH_DELIVER_BACKOFF_SEC", "10")
)

# Cap how much of an exception message is persisted to error_message.
_ERROR_MESSAGE_MAX_LEN = 2000

# Prefix for the per-call staging file the splice renders into before it is
# atomically renamed onto the candidate path. Dot-prefixed so a leftover from a
# hard-killed worker is hidden rather than mistaken for a deliverable.
_STAGING_PREFIX = ".staging-"

# How long a staging file must have been untouched before a later call treats
# it as debris rather than a live render. Comfortably past
# tasks.AUDIO_PATCH_JOB_TIMEOUT (3600s), so a sweep can never delete a file a
# concurrent worker is still writing — only one a killed worker abandoned.
# Clamped, not merely defaulted: file age is not a liveness proof, and the only
# thing that makes the sweep safe when two workers share a patch directory is
# that the threshold is comfortably past the job timeout. An env override of
# `0` would let a duplicate worker delete the other's just-created staging file
# and break its os.replace, so the floor is enforced here rather than trusted.
_STAGING_STALE_FLOOR_SEC = 7200.0
_STAGING_STALE_AFTER_SEC = max(
    _STAGING_STALE_FLOOR_SEC,
    float(os.getenv("AUDIO_PATCH_STAGING_STALE_SEC", "7200")),
)

# Slack when a requested bound is compared against a duration — the same
# value media.splice_audio_window uses, so the endpoint's 400 and the
# ffmpeg pass can never disagree about whether a window fits.
_BOUND_EPS_SEC = 0.05

# Run statuses a patch may be created against. Only `done`: a patch rewrites
# the run's delivered final, which is the one artifact a non-done run does not
# have. Same gate, and the same reasoning, as
# api_v2._RANGE_REGEN_ELIGIBLE_RUN_STATUSES.
AUDIO_PATCH_ELIGIBLE_RUN_STATUSES = {RunStatus.done}

# AICP-389: patch kinds. ``donor`` re-voices a window from a sibling run (the
# original feature, every row before AICP-389); ``same_run_lowpass`` is the
# autopilot's free brand fix — the window of the run's OWN audio low-passed
# (app.brand_check.render_candidate). A filter patch never renders: it
# promotes the candidate the operator approved, byte for byte.
KIND_DONOR = "donor"
KIND_SAME_RUN_LOWPASS = "same_run_lowpass"
PATCH_KINDS = (KIND_DONOR, KIND_SAME_RUN_LOWPASS)


def patch_kind(patch: RunAudioPatch) -> str:
    return getattr(patch, "kind", None) or KIND_DONOR


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class AudioPatchValidationError(ValueError):
    """A patch request that is invalid on its face — surfaced as a 400.

    Separate from MediaError (raised later, by the ffmpeg pass, and surfaced as
    a failed job) so the API can reject an impossible window before a row and a
    worker job exist for it.
    """


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


def _truncate_error(exc: Exception) -> str:
    """Render *exc* as a bounded-length string safe to store in error_message."""
    msg = str(exc)
    if len(msg) > _ERROR_MESSAGE_MAX_LEN:
        msg = msg[:_ERROR_MESSAGE_MAX_LEN] + "... (truncated)"
    return msg


def _sweep_stale_staging(out_dir: str, patch_id: str) -> None:
    """Delete staging files a hard-killed worker abandoned under this RUN's
    ``audio_patches/`` tree — not just this patch's own directory.

    The ``finally`` around the splice removes its own staging file, but a
    SIGKILL — which is what a redeploy does to an in-flight job in this project
    — never runs it, and nothing else ever revisits the directory: the recovery
    reaper only marks the row failed, it does not touch disk. Each leak is a
    whole video.

    Sweeping only ``out_dir`` would never collect any of them, which is the
    trap this docstring exists to record: ``failed`` is terminal here, so the
    operator's retry is a NEW patch with a NEW directory, and the killed one's
    debris sits in a directory nothing ever opens again. So the sweep walks the
    run's whole ``audio_patches/`` parent, one level down.

    Only files older than :data:`_STAGING_STALE_AFTER_SEC` are removed, and
    that threshold is floored well past the job timeout — so this can never
    delete a staging file some other worker is still writing, in this patch's
    directory or a sibling's. Failures are logged and ignored: housekeeping
    must never fail a patch.
    """
    root = os.path.dirname(os.path.normpath(out_dir))
    now = time.time()
    try:
        patch_dirs = [os.path.join(root, name) for name in os.listdir(root)]
    except OSError:
        return
    for patch_dir in patch_dirs:
        try:
            names = os.listdir(patch_dir)
        except OSError:
            continue  # not a directory, or vanished under us
        for name in names:
            if not name.startswith(_STAGING_PREFIX):
                continue
            path = os.path.join(patch_dir, name)
            try:
                if now - os.path.getmtime(path) < _STAGING_STALE_AFTER_SEC:
                    continue
                os.remove(path)
                log.info(
                    "patch_id=%s: removed stale staging file %s", patch_id, path
                )
            except OSError as exc:
                log.warning(
                    "patch_id=%s: could not remove stale staging file %s: %s",
                    patch_id, path, exc,
                )


def _resolve_result(run: Run, *, what: str) -> str:
    """The on-disk final video for *run*, or raise with a clear reason."""
    path = run.result_local_path
    if not path or not os.path.exists(path):
        raise AudioPatchValidationError(
            f"{what} run {run.id[:8]!r} has no result video on disk"
        )
    return path


def _known_duration(
    run: Run, path: str, override: Optional[float] = None
) -> float:
    """*run*'s result duration: *override* if given, else the cached column,
    else a probe.

    Mirrors create_tail_edit's reasoning: this runs on an operator click, not a
    poll, so one ffprobe is cheap next to creating a row and a worker job that
    would only fail a cycle later. The cached value is preferred because the
    delivery pipeline writes it from a probe of this very file.

    *override* exists so the API can resolve both durations BEFORE it takes the
    Run row lock and hand them in: ``media._run`` passes no timeout to
    ``subprocess.run``, so a hung ffprobe inside a locked window would hold an
    open transaction and that run's row lock indefinitely (prod runs with
    ``idle_in_transaction_session_timeout=0``). See
    ``api_v2.create_run_audio_patch``.
    """
    if override and override > 0:
        return float(override)
    cached = run.result_duration_sec
    if cached and cached > 0:
        return float(cached)
    return media_mod.probe(path).duration_sec


def validate_audio_patch_request(
    *,
    run: Run,
    donor: Run,
    target_start_sec: float,
    target_end_sec: float,
    donor_start_sec: float,
    crossfade_sec: float,
    target_duration_sec: Optional[float] = None,
    donor_duration_sec: Optional[float] = None,
) -> tuple[str, str]:
    """Check a patch request and return ``(target_path, donor_path)``.

    Everything checkable without running ffmpeg is checked here, so the
    operator gets a 400 naming the exact bound instead of a row that fails a
    minute later in a worker. Shared by the API (before the row exists) and by
    the worker (against the CURRENT state, which may have moved while the job
    sat in the queue). :func:`media.splice_audio_window` re-validates against
    the probed files regardless — this is the early, readable pass, not the
    authority.

    Raises
    ------
    AudioPatchValidationError
    """
    if donor.id == run.id:
        raise AudioPatchValidationError(
            "the donor must be a different run — a run cannot borrow its own audio"
        )
    donor_status = getattr(donor.status, "value", donor.status)
    if donor_status != RunStatus.done.value:
        raise AudioPatchValidationError(
            f"donor run {donor.id[:8]!r} is not done (status={donor_status})"
        )
    # Same project, enforced HERE and not only in the picker. web_v2's
    # _audio_patch_donor_runs already offers siblings only, but that is a UI
    # convenience, and this endpoint takes a run id straight from a form: a
    # pasted id from another project would otherwise splice unrelated speech
    # into a delivered creative, permanently and behind its existing Final
    # Creatives link. A donor is only meaningful when it speaks the same
    # script over the same footage, which is exactly what sharing a project
    # means here.
    if donor.project_id != run.project_id:
        raise AudioPatchValidationError(
            f"donor run {donor.id[:8]!r} belongs to a different project — a "
            "donor has to speak the same script over the same footage, so it "
            "must be a run of this run's own project"
        )

    target_path = _resolve_result(run, what="target")
    donor_path = _resolve_result(donor, what="donor")

    if target_start_sec < 0 or donor_start_sec < 0:
        raise AudioPatchValidationError("timestamps must not be negative")
    if crossfade_sec < 0:
        raise AudioPatchValidationError("crossfade must not be negative")

    window = target_end_sec - target_start_sec
    if window < media_mod.AUDIO_SPLICE_MIN_WINDOW_SEC:
        raise AudioPatchValidationError(
            f"the window is {window:.3f}s — it must be at least "
            f"{media_mod.AUDIO_SPLICE_MIN_WINDOW_SEC}s and end after it starts"
        )

    target_dur = _known_duration(run, target_path, target_duration_sec)
    if target_end_sec > target_dur + _BOUND_EPS_SEC:
        raise AudioPatchValidationError(
            f"the window ends at {target_end_sec:.2f}s but this run's video is "
            f"{target_dur:.2f}s long"
        )

    donor_dur = _known_duration(donor, donor_path, donor_duration_sec)
    donor_end = donor_start_sec + window
    if donor_end > donor_dur + _BOUND_EPS_SEC:
        raise AudioPatchValidationError(
            f"the donor would have to supply audio up to {donor_end:.2f}s "
            f"({window:.2f}s from {donor_start_sec:.2f}s) but run "
            f"{donor.id[:8]!r} is only {donor_dur:.2f}s long"
        )

    return target_path, donor_path


def create_audio_patch(
    *,
    run: Run,
    donor: Run,
    target_start_sec: float,
    target_end_sec: float,
    donor_start_sec: float,
    crossfade_sec: float,
    db: Session,
    target_duration_sec: Optional[float] = None,
    donor_duration_sec: Optional[float] = None,
) -> RunAudioPatch:
    """Validate and build (add + flush, but do NOT commit or enqueue) a patch.

    ``base_result_path`` is captured here, from the run's CURRENT result — that
    is the file the operator scrubbed to pick the window, and _promote refuses
    to promote onto anything else.
    """
    target_path, _donor_path = validate_audio_patch_request(
        run=run,
        donor=donor,
        target_start_sec=target_start_sec,
        target_end_sec=target_end_sec,
        donor_start_sec=donor_start_sec,
        crossfade_sec=crossfade_sec,
        target_duration_sec=target_duration_sec,
        donor_duration_sec=donor_duration_sec,
    )

    patch = RunAudioPatch(
        run_id=run.id,
        donor_run_id=donor.id,
        target_start_sec=target_start_sec,
        target_end_sec=target_end_sec,
        donor_start_sec=donor_start_sec,
        crossfade_sec=crossfade_sec,
        base_result_path=target_path,
        status=AudioPatchStatus.queued,
    )
    db.add(patch)
    db.flush()
    return patch


def create_lowpass_patch(
    *,
    run: Run,
    window: tuple[float, float],
    filter_params: dict,
    base_path: str,
    base_sha256: str,
    candidate_path: str,
    candidate_sha256: str,
    db: Session,
) -> RunAudioPatch:
    """Validate and build (add + flush, no commit, no enqueue) a filter patch.

    The caller holds the run-row lock (``api_v2._lock_run_for_result_mutation``)
    and has checked that nothing else is changing the run's result. Here the
    approved bytes are re-checked: the run still serves *base_path*, its bytes
    still hash to *base_sha256*, and the candidate still hashes to
    *candidate_sha256*. A mismatch is stale, never "close enough".
    """
    status = getattr(run.status, "value", run.status)
    if status != RunStatus.done.value:
        raise AudioPatchValidationError(f"the run is {status}, not done")
    if run.result_local_path != base_path:
        raise AudioPatchValidationError(
            "the run's result changed since the fix was rendered — the candidate is stale")
    if not base_path or not os.path.isfile(base_path) or _sha256(base_path) != base_sha256:
        raise AudioPatchValidationError(
            "the run's result bytes changed since the fix was rendered — the candidate is stale")
    if not candidate_path or not os.path.isfile(candidate_path) \
            or _sha256(candidate_path) != candidate_sha256:
        raise AudioPatchValidationError("the approved candidate file is missing or changed")
    start, end = float(window[0]), float(window[1])
    if not 0 <= start < end:
        raise AudioPatchValidationError("the window must end after it starts")
    patch = RunAudioPatch(
        run_id=run.id,
        kind=KIND_SAME_RUN_LOWPASS,
        donor_run_id=None,
        donor_start_sec=None,
        target_start_sec=start,
        target_end_sec=end,
        crossfade_sec=float(filter_params.get("ramp_sec") or 0.0),
        base_result_path=base_path,
        base_sha256=base_sha256,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        filter_params=dict(filter_params),
        status=AudioPatchStatus.queued,
    )
    db.add(patch)
    db.flush()
    return patch


def _active_range_attempt_id(session: Session, run_id: str) -> Optional[str]:
    from .models import RunSegmentRangeAttempt
    from .state_machine import RANGE_ATTEMPT_ACTIVE_STATUSES

    from sqlalchemy import select

    return session.execute(
        select(RunSegmentRangeAttempt.id).where(
            RunSegmentRangeAttempt.run_id == run_id,
            RunSegmentRangeAttempt.status.in_([s.value for s in RANGE_ATTEMPT_ACTIVE_STATUSES]),
        ).limit(1)
    ).scalar_one_or_none()


def _deliver_with_retry(
    gdrive: GDriveClient,
    *,
    path: str,
    file_id: Optional[str],
    folder_id: Optional[str],
    filename: str,
    patch_id: str,
) -> tuple[str, bool]:
    """Put the patched video on Drive; return ``(file_id, is_new_file)``.

    When the run already has a Drive result its CONTENT is replaced, keeping
    the id — that is what makes an already-exported Final Creatives row serve
    the fixed video instead of the broken one. A run with no Drive result yet
    (never delivered, or delivery failed) gets a normal upload instead, so the
    patch is still useful there.

    The caller guarantees at least one of *file_id* / *folder_id* — "there is
    nowhere to deliver to" is a configuration fact, not a transient error, and
    retrying it three times with backoff would only slow down the answer.

    Retries the whole transfer, like pipeline_tail._deliver_with_retry: both
    calls are resumable uploads, so a retry re-sends bytes rather than
    corrupting anything.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, AUDIO_PATCH_DELIVER_ATTEMPTS + 1):
        try:
            if file_id:
                gdrive.update_file_content(file_id, path)
                return file_id, False
            result = gdrive.upload_file(path, folder_id, filename)
            new_id = result.get("id")
            if not new_id:
                raise RuntimeError("Drive upload returned no file id")
            return new_id, True
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Drive delivery attempt %d/%d failed for audio patch %s: %s",
                attempt, AUDIO_PATCH_DELIVER_ATTEMPTS, patch_id, exc,
            )
            if attempt < AUDIO_PATCH_DELIVER_ATTEMPTS:
                time.sleep(AUDIO_PATCH_DELIVER_BACKOFF_SEC * attempt)
    raise last_exc  # type: ignore[misc]


def _promote(
    session: Session,
    run: Run,
    patch: RunAudioPatch,
    *,
    gdrive: GDriveClient,
    filename: str,
) -> None:
    """Make the candidate the run's result — Drive first, then the row, in one
    locked transaction that also marks the patch ``done``.

    Locks the Run row before checking anything on it. The DB-level partial
    unique index stops a SECOND patch being active on this run, but not the
    SAME patch being processed by two worker sessions (a duplicate enqueue, or
    a recovery re-enqueue racing a still-running job — both have happened to
    other jobs in this codebase). ``session.refresh(..., with_for_update=True)``
    re-reads the row under a real lock and overwrites this object's attributes
    with what is currently persisted, so a loser sees the winner's already
    re-pointed ``result_local_path``, fails the staleness check below and
    refuses instead of overwriting it. A no-op on SQLite, which has no
    row-level locking; the guarantee is against PostgreSQL, this feature's
    production database.

    **The Drive update happens inside that lock, deliberately.** The whole
    point of the feature is that the run's local result and the bytes behind
    its Drive id say the same thing; doing the transfer before the lock would
    mean a patch that then loses the staleness race has already replaced a
    creative it is about to disown. Holding a row lock across a ~20 MB upload
    is affordable precisely because the index makes contention on it
    near-impossible. It also depends on the database not killing a transaction
    that sits idle mid-upload: verified 2026-09-07 that prod's
    ``idle_in_transaction_session_timeout``, ``statement_timeout`` and
    ``lock_timeout`` are all 0 (disabled). Setting any of them would need this
    ordering revisited.

    Residual risk, accepted and not worked around: a worker killed in the
    instant between Drive acknowledging the update and this commit landing
    leaves Drive holding the patched video while the Run still points at the
    old local file. The recovery reaper will mark the patch ``failed``; re-
    running the same window fixes it, because replacing the same content twice
    is idempotent. The alternative — promoting first and uploading after —
    trades this for the strictly worse failure of the sheet's link serving the
    broken audio while the app claims it is fixed.
    """
    session.refresh(run, with_for_update=True)
    # AICP-389 gate: lock the patch row too (run first, then patch) and make
    # sure it is still THIS attempt's `delivering` row BEFORE Drive is touched.
    # The recovery reaper fails an orphan-looking row with a version-checked
    # UPDATE: without this lock it could commit `failed` mid-upload, and this
    # commit would then lose on version_id after Drive already holds the
    # candidate. With it, the reaper's UPDATE waits and then matches nothing.
    session.refresh(patch, with_for_update=True)
    patch_status = getattr(patch.status, "value", patch.status)
    if patch_status != AudioPatchStatus.delivering.value:
        raise RuntimeError(
            f"refusing to promote: the patch is {patch_status!r}, not delivering "
            "(reaped or taken over) — the delivered result is left untouched"
        )
    if run.status != RunStatus.done:
        raise RuntimeError(
            f"refusing to promote: run status changed to {run.status!r} while "
            "this patch was in flight — the delivered result is left untouched"
        )
    if run.result_local_path != patch.base_result_path:
        raise RuntimeError(
            "refusing to promote: the run's result changed since this patch "
            "was created (another audio patch or a range attempt already "
            "promoted) — this candidate was NOT applied; check the current "
            "result and start a new patch if a correction is still needed"
        )
    if patch_kind(patch) == KIND_SAME_RUN_LOWPASS:
        # AICP-389: the approved bytes, re-checked under the lock and BEFORE
        # the Drive update: the base still hashes as approved, the file to
        # deliver is the approved candidate, and no range attempt is active.
        if not os.path.isfile(run.result_local_path or "") \
                or _sha256(run.result_local_path) != patch.base_sha256:
            raise RuntimeError(
                "refusing to promote: the run's result bytes changed since the "
                "fix was approved — this candidate was NOT applied")
        if not patch.output_path or _sha256(patch.output_path) != patch.candidate_sha256:
            raise RuntimeError(
                "refusing to promote: the file to deliver is not the approved candidate")
        attempt = _active_range_attempt_id(session, run.id)
        if attempt is not None:
            raise RuntimeError(
                f"refusing to promote: range attempt {attempt} is active on this run")

    folder_id = run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
    if run.result_gdrive_file_id or folder_id:
        file_id, is_new = _deliver_with_retry(
            gdrive,
            path=patch.output_path,
            file_id=run.result_gdrive_file_id,
            folder_id=folder_id,
            filename=filename,
            patch_id=patch.id,
        )
        patch.gdrive_file_id = file_id
        if is_new:
            run.result_gdrive_file_id = file_id
    else:
        # No Drive file to repair and nowhere to put a new one — the local
        # result is still worth fixing, and there is no divergence to create
        # because there is nothing on Drive to diverge from. Mirrors
        # pipeline_tail's "no folder_id -> skip delivery, still done".
        file_id, is_new = None, False
        log.warning(
            "audio patch %s: run %s has no Drive result and no folder — "
            "promoting the patched video locally only",
            patch.id, run.id,
        )

    run.result_local_path = patch.output_path
    if patch.result_duration_sec is not None:
        run.result_duration_sec = patch.result_duration_sec

    patch.promoted_at = datetime.now(timezone.utc)
    patch.error_message = None
    transition(patch, AudioPatchStatus.done)
    session.commit()
    log.info(
        "audio patch %s: PROMOTED onto run %s — window=[%.3f, %.3f) from "
        "%s %s@%s, drive_file=%s (%s)",
        patch.id, run.id, patch.target_start_sec, patch.target_end_sec,
        patch_kind(patch), patch.donor_run_id, patch.donor_start_sec, file_id,
        "new upload" if is_new
        else ("content replaced in place" if file_id else "no Drive delivery"),
    )


def _stage_approved_candidate(session: Session, run: Run, patch: RunAudioPatch) -> None:
    """Copy the approved candidate into the patch directory and verify it.

    A filter patch never renders here: the bytes the operator heard in the
    preview's full counterpart are the bytes delivered. The copy lands under a
    staging name and is renamed atomically (the duplicate-worker reasoning of
    the donor splice below); its hash must still be the approved one. No size
    enforcement either: re-encoding would change the approved bytes.
    """
    import shutil

    if run.result_local_path != patch.base_result_path:
        raise RuntimeError(
            "the run's result changed after this fix was approved — start a new "
            "check against the current result")
    if not patch.candidate_path or not os.path.isfile(patch.candidate_path):
        raise RuntimeError("the approved candidate file is gone")
    out_dir = audio_patch_dir(patch.id, run.id, run.project_id)
    _sweep_stale_staging(out_dir, patch.id)
    output_path = os.path.join(out_dir, f"{run.id[:8]}_brandfix_{patch.id[:8]}.mp4")
    staging_path = os.path.join(out_dir, f"{_STAGING_PREFIX}{uuid.uuid4().hex}.mp4")
    try:
        shutil.copyfile(patch.candidate_path, staging_path)
        if _sha256(staging_path) != patch.candidate_sha256:
            raise RuntimeError("the approved candidate changed while it was copied")
        os.replace(staging_path, output_path)
    finally:
        if os.path.exists(staging_path):
            try:
                os.remove(staging_path)
            except OSError:
                log.warning("patch_id=%s: could not remove staging file %s", patch.id, staging_path)
    patch.output_path = output_path
    patch.result_duration_sec = media_mod.probe(output_path).duration_sec
    session.commit()


def process_audio_patch(
    patch_id: str, *, gdrive: Optional[GDriveClient] = None
) -> Optional[PublishedFinal]:
    """
    Splice a donor run's audio window into this run's final and promote it.

    Returns the committed :class:`PublishedFinal` after this call's own
    promotion (AICP-299 §4), else ``None`` (already done); the RQ wrapper
    makes the knowledge publication link from it.

    Transitions
    -----------
    queued -> processing -> delivering -> done
    (on error: -> failed, error_message set, exception re-raised)
    """
    t_start = time.monotonic()
    log.info("process_audio_patch start: patch_id=%s", patch_id)

    with get_session() as session:
        patch: RunAudioPatch = session.get(RunAudioPatch, patch_id)
        if patch is None:
            raise ValueError(f"RunAudioPatch not found: {patch_id}")

        try:
            # Orphan resume: any non-queued starting status means a prior
            # attempt was interrupted mid-flight (mirrors
            # pipeline_tail.process_tail_edit). `done` is terminal and has no
            # edge back, so a promoted patch is never reprocessed — it returns
            # below instead.
            status_val = getattr(patch.status, "value", patch.status)
            if status_val == AudioPatchStatus.done.value:
                log.info(
                    "process_audio_patch: patch %s is already done — nothing "
                    "to do", patch_id,
                )
                return
            if patch.status != AudioPatchStatus.queued:
                log.info(
                    "process_audio_patch: patch %s is in status=%r (not "
                    "queued) — resetting to queued for orphan resume",
                    patch_id, patch.status,
                )
                # Assigned, not transitioned: this enum has no edge back to
                # `queued` from anywhere (unlike TailEditStatus, which allows
                # re-running a finished edit), and going via `failed` would
                # write a failure that never happened. `done` already returned
                # above, so this only ever rewinds a genuinely orphaned row.
                patch.status = AudioPatchStatus.queued
                patch.updated_at = datetime.now(timezone.utc)
                session.commit()

            transition(patch, AudioPatchStatus.processing)
            session.commit()

            run: Run = session.get(Run, patch.run_id)
            if run is None:
                raise ValueError(f"Run not found: {patch.run_id}")
            if patch_kind(patch) == KIND_SAME_RUN_LOWPASS:
                _stage_approved_candidate(session, run, patch)
                transition(patch, AudioPatchStatus.delivering)
                session.commit()
                if gdrive is None:
                    gdrive = _default_gdrive()
                _promote(session, run, patch, gdrive=gdrive,
                         filename=os.path.basename(patch.output_path))
                log.info("process_audio_patch done (%s): patch_id=%s total=%.1fs",
                         KIND_SAME_RUN_LOWPASS, patch_id, time.monotonic() - t_start)
                return PublishedFinal(run_id=run.id, project_id=run.project_id,
                                      path=patch.output_path, kind="audio",
                                      publication_id=f"audio_patch:{patch.id}")
            donor: Run = session.get(Run, patch.donor_run_id)
            if donor is None:
                raise ValueError(
                    f"donor run {patch.donor_run_id!r} no longer exists"
                )

            # Re-validate against the CURRENT state, not create time: the donor
            # may have been re-run or the target re-pointed while this sat in
            # the queue.
            target_path, donor_path = validate_audio_patch_request(
                run=run,
                donor=donor,
                target_start_sec=patch.target_start_sec,
                target_end_sec=patch.target_end_sec,
                donor_start_sec=patch.donor_start_sec,
                crossfade_sec=patch.crossfade_sec,
            )
            if target_path != patch.base_result_path:
                raise RuntimeError(
                    "the run's result changed after this patch was created — "
                    "the window was picked against a different video; start a "
                    "new patch against the current result"
                )

            # ---------------------------------------------------------------
            # Splice
            # ---------------------------------------------------------------
            t_encode = time.monotonic()
            out_dir = audio_patch_dir(patch.id, run.id, run.project_id)
            _sweep_stale_staging(out_dir, patch_id)
            filename = f"{run.id[:8]}_audiofix_{patch.id[:8]}.mp4"
            output_path = os.path.join(out_dir, filename)

            # Render to a per-call staging name and publish it with an atomic
            # rename, never straight to output_path. The candidate path is
            # derived from the patch id alone, so two workers on the SAME patch
            # (a duplicate enqueue, or an orphan resume racing a job that is
            # still alive) would otherwise both point ffmpeg's `-y` at one
            # file: one truncates it while the other is uploading, and half a
            # file replaces an already-delivered creative on Drive. os.replace
            # is atomic and leaves an already-open reader on the old inode, so
            # the uploader always sees one complete video — whichever of the
            # two wrote it, both being the same splice of the same inputs.
            # version_id only catches that race at COMMIT time, which is after
            # the upload; this is what stops it before.
            # Keeps the .mp4 suffix: ffmpeg picks the output container from the
            # extension, and a ".part" name dies with "Unable to find a
            # suitable output format". Dot-prefixed so a stray one is hidden
            # and matches _STAGING_GLOB below.
            staging_path = os.path.join(
                out_dir, f"{_STAGING_PREFIX}{uuid.uuid4().hex}.mp4"
            )
            try:
                duration = media_mod.splice_audio_window(
                    target_path,
                    donor_path,
                    staging_path,
                    base_start=patch.target_start_sec,
                    base_end=patch.target_end_sec,
                    donor_start=patch.donor_start_sec,
                    crossfade_sec=patch.crossfade_sec,
                )
                os.replace(staging_path, output_path)
            finally:
                # A failed render leaves no half-written .part lying around.
                if os.path.exists(staging_path):
                    try:
                        os.remove(staging_path)
                    except OSError:
                        log.warning(
                            "patch_id=%s: could not remove staging file %s",
                            patch_id, staging_path,
                        )
            # Normally a no-op — the picture is stream-copied, so the only size
            # change is the re-encoded audio track. Never fail the patch over
            # it: an oversized deliverable still beats a mispronounced one.
            try:
                media_mod.enforce_size_limit(output_path)
            except Exception as exc:
                log.warning(
                    "patch_id=%s: could not shrink %s to the delivery size "
                    "cap: %s", patch_id, output_path, exc,
                )
            patch.output_path = output_path
            patch.result_duration_sec = duration
            session.commit()
            log.info(
                "patch_id=%s encode_phase=%.1fs -> %s (%.3fs)",
                patch_id, time.monotonic() - t_encode, output_path, duration,
            )

            # ---------------------------------------------------------------
            # Deliver + promote (one locked transaction — see _promote)
            # ---------------------------------------------------------------
            transition(patch, AudioPatchStatus.delivering)
            session.commit()

            if gdrive is None:
                gdrive = _default_gdrive()
            _promote(session, run, patch, gdrive=gdrive, filename=filename)

            log.info(
                "process_audio_patch done: patch_id=%s total=%.1fs",
                patch_id, time.monotonic() - t_start,
            )
            return PublishedFinal(run_id=run.id, project_id=run.project_id, path=patch.output_path,
                                  kind="audio", publication_id=f"audio_patch:{patch.id}")

        except Exception as exc:
            log.exception("process_audio_patch failed: patch_id=%s", patch_id)
            session.rollback()
            patch = session.get(RunAudioPatch, patch_id)
            if patch is not None:
                if patch.status not in (
                    AudioPatchStatus.done, AudioPatchStatus.failed,
                ):
                    try:
                        transition(patch, AudioPatchStatus.failed)
                    except Exception:
                        patch.status = AudioPatchStatus.failed
                patch.error_message = _truncate_error(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise
