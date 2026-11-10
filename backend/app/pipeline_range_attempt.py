"""
pipeline_range_attempt.py — run-local swap-segment range regeneration.

One :class:`~app.models.RunSegmentRangeAttempt` is one candidate revision of a
SINGLE swap segment's run-local time range: the operator moves the shared
boundary with an adjacent `keep` segment, and this module cuts the candidate
swap (and any re-cut `keep` neighbour) from the ORIGINAL project source,
generates it, re-stitches a candidate final video, and delivers it — all
against this row's own fields and its own working directory
(``storage.range_attempt_dir``). The Run's active ``segment_snapshot`` and the
targeted ``RunSegment``'s fields are left completely untouched until
:func:`poll_range_attempt_once` reaches a successful `done`, at which point it
promotes the candidate's fields onto them in one commit. A `failed` attempt
never touches the Run — see docs/run-local-segment-range-regeneration.md.

Shape mirrors :mod:`app.pipeline_segment_image` (submit / poll-once, no run-
wide Redis lock — serialised by api_v2's one-attempt-per-run 409 gate instead)
but the create-task/poll-download cycle is a full swap-segment video
generation (reusing :func:`pipeline_v2._create_swap_task`, the same call the
main pipeline makes) followed by a candidate stitch + deliver, since a range
change also moves an adjacent keep clip's cut point.

Two-phase, like segment-image regeneration, so the worker never sleeps:

    submit_range_attempt(attempt_id)      creates (or resumes) the kie.ai task
    poll_range_attempt_once(attempt_id)   one round of status-check; on
                                           success also stitches, delivers and
                                           PROMOTES. The caller (app/tasks.py)
                                           re-enqueues itself while pending.

Boundary validation (:func:`validate_range_request`) is a pure function over
primitive data — no DB/network/ffmpeg — so it is safe to call synchronously
from the API request handler, before anything is spent.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from . import ai_models
from .knowledge import capture as kcap
from .knowledge.publication import PublishedFinal
from . import media as media_mod
from . import spend as spend_mod
from .db import get_session
from .gdrive_client import GDriveClient, GDriveFileNotFound
from .kie_client import KieClient, get_shared_client
from .models import (
    LibraryClip,
    Run,
    RunSegment,
    RunSegmentRangeAttempt,
    VideoProject,
)
from .pipeline_common import resolve_reference_urls
from .pipeline_v2 import (
    _create_swap_task,
    _deliver_in_place_with_retry,
    _deliver_with_retry,
    _parse_cost_credits,
    _parse_result_url,
    _result_too_short,
    generate_audio_for_run,
)
from .state_machine import RangeAttemptStatus, RunStatus, SegmentStatus, transition
from .storage import keep_clip_path, range_attempt_dir, run_results_dir

log = logging.getLogger(__name__)

# Same bound used everywhere else source/candidate durations are compared
# against a stated value (RunSegment.error_message truncation aside) — the UI
# asks for 0.1s precision, so anything smaller than this is float noise, not
# an operator-intended move.
_EPS_SEC = 0.02

# Same tolerance _submit_swap_segment_isolated uses for its own min/max clip
# checks — sub-frame at any fps this pipeline handles.
_MODEL_DURATION_EPS_SEC = 0.05

_ERROR_MESSAGE_MAX_LEN = 2000

# Lease duration for a phase claim (see _try_claim_phase).
#
# MUST NOT be shorter than the RQ job_timeout both submit_range_attempt and
# poll_range_attempt_once run under (tasks.RANGE_ATTEMPT_POLL_JOB_TIMEOUT,
# 3600s at the time of writing — kept in sync by hand, since app.tasks
# cannot be imported here: it imports THIS module's functions, deferred
# inside its RQ-callable wrappers, specifically to avoid a cycle; a test
# asserts the relationship holds). RQ enforces job_timeout as a hard kill —
# a worker that overruns it is forcibly terminated — so sizing the lease to
# job_timeout PLUS a safety margin makes it PROVABLE, not merely assumed,
# that a genuinely still-running, healthy worker can never have its own
# claim expire out from under it. This is the exact same relationship
# tasks._RUN_LOCK_TTL_SEC already uses for the main pipeline's Redis lock
# (PROCESS_JOB_TIMEOUT + 300) — a proven pattern in this codebase, not a
# new one. No heartbeat/renewal loop is needed as a result: the lease is
# never "extended" mid-flight, it is simply long enough from the start.
#
# An earlier version of this constant (900s) was shorter than the 3600s job
# timeout — found by an independent audit: a finalize phase whose
# stitch+Drive-delivery genuinely took longer than the lease would let a
# duplicate poller (finding the "expired" claim, but the original worker
# very much still alive) wrongly take over mid-flight, reopening concurrent
# download/stitch/delivery/promotion and risking a duplicate Drive upload.
# The floor below makes that misconfiguration structurally impossible even
# if RANGE_ATTEMPT_CLAIM_LEASE_SEC is set too low in the environment.
#
# Recovering an ATTEMPT WHOSE OWNER HAS ACTUALLY CRASHED does not wait for
# this (now much longer) lease to expire on its own: recovery.py's periodic
# reaper (~5-8 minutes, driven by RQ job liveness, not by this lease) is the
# real backstop in production — see that module's docstring on
# _reap_orphaned_derived_creatives for the safe-resume-vs-fail split it
# performs. This module's own expiry check is defense in depth for the case
# that reaper does not run (disabled, Redis down, or a future re-invocation
# it does not model) — not the primary recovery mechanism.
_RANGE_ATTEMPT_JOB_TIMEOUT_SEC = 3600  # tasks.RANGE_ATTEMPT_POLL_JOB_TIMEOUT
_CLAIM_LEASE_SEC = max(
    int(os.environ.get("RANGE_ATTEMPT_CLAIM_LEASE_SEC", str(_RANGE_ATTEMPT_JOB_TIMEOUT_SEC + 300))),
    _RANGE_ATTEMPT_JOB_TIMEOUT_SEC + 60,
)

# Surfaced verbatim as RunSegmentRangeAttempt.error_message when a later call
# discovers a submit-phase claim expired after kie.ai's paid createTask call
# may already have been made (submit_billed_at is set) but no
# seedance_task_id was ever persisted — see submit_range_attempt and the
# module docstring below for why this can never be auto-retried.
def _ambiguous_submit_error(billed_at: Optional[datetime]) -> str:
    return (
        "Submission left in an AMBIGUOUS state: a previous worker committed "
        "submit_billed_at (about to call, or just called, kie.ai's paid "
        "createTask) but crashed before persisting a seedance_task_id, and "
        "its claim has since expired. kie.ai's jobs API has no idempotency "
        "key or lookup-by-reference, so it cannot be determined here "
        "whether a billed task was actually created. This attempt will NOT "
        "be retried automatically to avoid risking a second billed "
        f"generation for the same segment. Manually check the kie.ai "
        f"dashboard/account activity around {billed_at} for an unexpected "
        "task before starting a new range attempt for this segment."
    )


def _truncate_error(exc: Exception) -> str:
    msg = str(exc)
    if len(msg) > _ERROR_MESSAGE_MAX_LEN:
        msg = msg[:_ERROR_MESSAGE_MAX_LEN] + "... (truncated)"
    return msg


# ---------------------------------------------------------------------------
# Per-phase ownership claim — see RunSegmentRangeAttempt.claim_owner's
# docstring in models.py for the full rationale. Reused by BOTH the submit
# phase (submit_range_attempt) and the finalize phase
# (_handle_terminal_kie_state) since exactly one of them is ever active for
# a given attempt at a time (an attempt's status must reach `generating`
# before the finalize phase can begin, and the submit phase never runs
# again after that) — one pair of columns is enough for both.
# ---------------------------------------------------------------------------


def _is_claim_live(attempt: RunSegmentRangeAttempt, now: datetime) -> bool:
    """True if *attempt* currently has an unexpired claim — i.e. some OTHER
    call (this function is only ever consulted before a call takes its own
    claim) is actively working the current phase and must not be raced."""
    if not attempt.claim_owner or not attempt.claim_expires_at:
        return False
    expires = attempt.claim_expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > now


def _try_claim_phase(session: Session, attempt: RunSegmentRangeAttempt) -> Optional[str]:
    """Attempt to claim exclusive ownership of *attempt*'s current phase in
    one short, immediately-committed transaction, before this call does any
    file or external-client side effect.

    Returns a fresh opaque claim token on success — the claim is already
    durably committed by the time this returns, and no PostgreSQL row lock
    or open transaction is held across the ffmpeg/network work the caller
    is about to do. Returns ``None`` only for the BENIGN case: this call's
    own (possibly slightly stale, but not yet proven wrong) in-memory view
    shows another owner genuinely still holding a live, non-expired claim —
    exit without touching any file or external client and report "pending",
    since that other owner is presumably still working.

    Deliberately does NOT catch a ``StaleDataError`` raised by the claim
    commit itself — that means THIS call's premise was already stale (the
    row moved on further than this call could see, possibly all the way to
    `done`/`failed`), not merely "someone else is currently working it", and
    a caller must not conflate the two: swallowing it into "pending" would
    make a duplicate poller report a still-in-progress attempt that had, in
    fact, ALREADY finished (proved by
    ``test_duplicate_poll_of_the_same_attempt_never_clobbers_the_others_committed_done``).
    Let it propagate so the caller's existing StaleDataError handling
    re-reads the row and reports its TRUE current state.

    Callers with phase-specific "is taking over an EXPIRED claim actually
    safe" logic (submit_range_attempt's submit_billed_at check) must run
    that check BEFORE calling this, using ``_is_claim_live`` to decide
    whether there is even a takeover to reason about, and must decide for
    themselves how to handle a StaleDataError from this call.
    """
    now = datetime.now(timezone.utc)
    if _is_claim_live(attempt, now):
        return None
    token = uuid.uuid4().hex
    attempt.claim_owner = token
    attempt.claim_expires_at = now + timedelta(seconds=_CLAIM_LEASE_SEC)
    session.commit()
    return token


def _claim_dir(attempt_dir: str, token: str) -> str:
    """Claim-scoped working directory for one phase-claim attempt.

    Nested under the attempt's own directory (never shared across attempt
    ids) and namespaced by the claim token (never shared across two
    acquisitions of the same attempt's claim, whether concurrent racers or a
    later takeover) — so no two callers, ever, write the same path. A
    published field (``local_clip_path``/``local_result_path``/
    ``final_local_path``) simply points INTO the winning claim's directory;
    a losing/superseded claim's directory is deleted wholesale by
    ``_cleanup_claim`` and can never be a winner's.
    """
    path = os.path.join(attempt_dir, "claims", token)
    os.makedirs(path, exist_ok=True)
    return path


def _staging_path(claim_dir: str, dst: str) -> str:
    """A claim-scoped staging path for *dst*'s eventual atomic publish
    (``os.replace``) — keeps *dst*'s own extension (ffmpeg/media_mod infer
    the output container from it; a bare ``.tmp`` suffix breaks that, since
    ffmpeg then sees an unrecognised container and refuses to write) while
    staying uniquely named within *claim_dir*, which is itself unique per
    claim — so no two claims of the same attempt, nor two different cache
    entries of the same claim, ever collide on this path."""
    root, ext = os.path.splitext(os.path.basename(dst))
    return os.path.join(claim_dir, f"_staged_{root}{ext}")


def _cleanup_claim(claim_dir: str) -> None:
    """Best-effort delete of one (and only one) claim's own working
    directory — used only when THIS call's claim has been superseded
    (StaleDataError on a later commit). Scoped to that single claim
    directory, so this can never remove a winner's — or any other claim's —
    artifacts."""
    shutil.rmtree(claim_dir, ignore_errors=True)


class _ClaimLost(Exception):
    """Raised by :func:`_assert_still_owner` when an explicit re-check finds
    this call's ``claim_token`` no longer matches the row's CURRENT
    ``claim_owner`` — this worker's ownership was superseded by another
    claimant since it last committed. Handled identically to
    ``StaleDataError`` by every caller in this module: abandon this call's
    remaining work, clean up only its own claim directory, and let the
    winner's outcome stand (see the ``except (StaleDataError, _ClaimLost)``
    sites in ``_handle_terminal_kie_state``/``submit_range_attempt``).

    A dedicated exception rather than raising ``StaleDataError`` directly:
    that class is SQLAlchemy's own flush/version-mismatch signal, raised
    automatically by the ORM — reusing it for a manually-issued check would
    be misleading about where the failure actually came from, even though
    both are handled the same way downstream.
    """


def _assert_still_owner(session: Session, attempt_id: str, claim_token: str) -> None:
    """Actively re-verify THIS caller still owns *attempt_id*'s claim, by
    re-querying the row's CURRENT ``claim_owner`` — not the in-memory copy,
    which cannot observe a takeover committed by a DIFFERENT session/
    connection without an explicit re-read (SQLAlchemy's identity map only
    reflects what THIS session itself last read or wrote). Under
    PostgreSQL's default READ COMMITTED isolation, a fresh ``SELECT`` like
    this one always sees every OTHER session's commits that landed before
    it runs, regardless of what this session's own (unrelated) open
    transaction holds elsewhere.

    Raises :class:`_ClaimLost` if superseded.

    This is DEFENSE IN DEPTH on top of ``_CLAIM_LEASE_SEC`` being sized to
    RQ's own job_timeout — which already makes a genuine takeover of a
    still-alive, healthy owner provably impossible under normal operation
    (see that constant's docstring). It guards the residual risk of clock
    skew between processes, a manually-edited claim row, or an
    as-yet-unknown bug — "do not rely only on a normally-finishes-quickly
    assumption" applies here too. Called before every remaining external or
    publish side effect: downloading, re-cutting a moved `keep` neighbour,
    publishing into the shared keep-cache, stitching, delivering to Drive,
    calling kie.ai's createTask, and promoting.
    """
    current = session.execute(
        select(RunSegmentRangeAttempt.claim_owner).where(
            RunSegmentRangeAttempt.id == attempt_id
        )
    ).scalar_one_or_none()
    if current != claim_token:
        raise _ClaimLost(
            f"attempt {attempt_id}: claim_owner is now {current!r}, no "
            f"longer this call's token {claim_token!r} — ownership was "
            "superseded; abandoning this call's remaining work"
        )


class RangeValidationError(ValueError):
    """An operator-readable reason a candidate range cannot be accepted.

    Raised only from pure validation (no media cut, upload, external call, Run
    state transition, or candidate artifact write has happened yet) — the API
    layer turns this into a 400 before anything is spent.
    """


# ---------------------------------------------------------------------------
# Pure boundary math — no DB, no I/O. See docs/run-local-segment-range-
# regeneration.md for the full rule set with worked examples.
# ---------------------------------------------------------------------------


def compute_candidate_snapshot(
    base_snapshot: list[dict],
    target_segment_def_id: str,
    new_start: float,
    new_end: float,
    source_duration_sec: float,
) -> tuple[list[dict], Optional[str], Optional[str]]:
    """Validate a candidate (new_start, new_end) for the target swap entry.

    Returns ``(candidate_snapshot, moved_prev_id, moved_next_id)`` —
    *candidate_snapshot* is a full deep copy of *base_snapshot* with the
    target entry's bounds updated and, for whichever side(s) actually moved,
    the immediately adjacent entry's shared boundary updated to match.
    *moved_prev_id* / *moved_next_id* are the ``segment_def_id`` of the
    `keep` neighbour(s) that were re-bounded (``None`` when that side did not
    change) — the caller re-cuts exactly those, and only those.

    Raises :class:`RangeValidationError` for every rejection this feature
    defines: non-finite/reversed/out-of-source-range bounds, a locked
    boundary (missing or non-`keep` neighbour — this is what makes
    `swap -> swap` impossible), and a neighbour fully consumed by the move.
    Does NOT validate against the selected model's duration limits — see
    :func:`validate_model_duration`, kept separate because it needs the run's
    model, not just the timeline.
    """
    if not math.isfinite(new_start) or not math.isfinite(new_end):
        raise RangeValidationError("start/end must be finite numbers")
    if new_end - new_start <= _EPS_SEC:
        raise RangeValidationError(
            f"end ({new_end:.2f}s) must be after start ({new_start:.2f}s)"
        )
    if new_start < -_EPS_SEC:
        raise RangeValidationError(f"start ({new_start:.2f}s) cannot be before 0")
    if new_end > source_duration_sec + _EPS_SEC:
        raise RangeValidationError(
            f"end ({new_end:.2f}s) cannot be after the source's duration "
            f"({source_duration_sec:.2f}s)"
        )
    new_start = max(0.0, new_start)
    new_end = min(source_duration_sec, new_end)

    ordered = sorted(base_snapshot, key=lambda e: e["index"])
    pos = next(
        (i for i, e in enumerate(ordered) if e.get("id") == target_segment_def_id),
        None,
    )
    if pos is None:
        raise RangeValidationError(
            "this segment is not part of the run's active timeline"
        )
    target = ordered[pos]
    if target.get("action") != "swap":
        raise RangeValidationError("only a swap segment's range can be regenerated")

    prev_entry = ordered[pos - 1] if pos > 0 else None
    next_entry = ordered[pos + 1] if pos < len(ordered) - 1 else None

    start_changed = abs(new_start - float(target["start_sec"])) > _EPS_SEC
    end_changed = abs(new_end - float(target["end_sec"])) > _EPS_SEC

    if not start_changed and not end_changed:
        raise RangeValidationError(
            "the requested range is the same as the segment's current range"
        )

    if start_changed:
        if prev_entry is None or prev_entry.get("action") != "keep":
            raise RangeValidationError(
                "cannot move the start boundary: the previous timeline entry "
                "is not a `keep` segment (moving a boundary shared with "
                "another swap segment is not supported)"
            )
    if end_changed:
        if next_entry is None or next_entry.get("action") != "keep":
            raise RangeValidationError(
                "cannot move the end boundary: the next timeline entry is "
                "not a `keep` segment (moving a boundary shared with "
                "another swap segment is not supported)"
            )

    candidate = copy.deepcopy(ordered)
    candidate[pos]["start_sec"] = new_start
    candidate[pos]["end_sec"] = new_end
    moved_prev_id = None
    moved_next_id = None
    if start_changed:
        candidate[pos - 1]["end_sec"] = new_start
        moved_prev_id = candidate[pos - 1]["id"]
    if end_changed:
        candidate[pos + 1]["start_sec"] = new_end
        moved_next_id = candidate[pos + 1]["id"]

    # Re-validate the WHOLE candidate timeline is still an ordered, gap-free,
    # overlap-free partition of [0, source_duration_sec] — this is what
    # catches a `keep` neighbour being fully (or negatively) consumed by the
    # move, without hand-deriving a separate inequality for every case.
    for i, entry in enumerate(candidate):
        s, e = float(entry["start_sec"]), float(entry["end_sec"])
        if not math.isfinite(s) or not math.isfinite(e):
            raise RangeValidationError("resulting timeline has a non-finite bound")
        if e - s <= _EPS_SEC:
            label = "the segment" if i == pos else "an adjacent keep segment"
            raise RangeValidationError(
                f"this move would leave {label} with zero or negative "
                "duration — pick a smaller move"
            )
        if i == 0 and abs(s - 0.0) > _EPS_SEC:
            raise RangeValidationError(
                "resulting timeline would not start at 0 — internal error"
            )
        if i > 0 and abs(s - float(candidate[i - 1]["end_sec"])) > _EPS_SEC:
            raise RangeValidationError(
                "this move would leave a gap or overlap in the timeline"
            )
        if i == len(candidate) - 1 and abs(e - source_duration_sec) > _EPS_SEC:
            raise RangeValidationError(
                "resulting timeline would not cover the full source — "
                "internal error"
            )

    return candidate, moved_prev_id, moved_next_id


def validate_model_duration(model: Optional[str], new_start: float, new_end: float) -> None:
    """Raise :class:`RangeValidationError` if the candidate swap duration is
    outside the selected model's effective input constraints.

    Deliberately a hard rejection, not a pad-to-fit like the main submit path
    (``_submit_swap_segment_isolated``'s pre/post-roll padding): the operator
    asked for an EXACT range, and padding would silently pull in footage
    outside it. See docs/run-local-segment-range-regeneration.md.
    """
    dur = new_end - new_start
    min_c = ai_models.min_clip_sec(model)
    max_c = ai_models.max_clip_sec(model)
    if dur < min_c - _MODEL_DURATION_EPS_SEC:
        raise RangeValidationError(
            f"duration {dur:.2f}s is shorter than the {min_c:.1f}s minimum "
            f"{ai_models.spec_for(model).label} accepts — widen the range"
        )
    if dur > max_c + _MODEL_DURATION_EPS_SEC:
        raise RangeValidationError(
            f"duration {dur:.2f}s exceeds the {max_c:.1f}s maximum "
            f"{ai_models.spec_for(model).label} accepts — narrow the range"
        )


def validate_stitch_plan_coherence(stitch_plan: Optional[list]) -> None:
    """Raise if *stitch_plan* has a shape this feature cannot safely re-stitch.

    Range regeneration never adds, removes or reorders a segment_def_id or
    library entry — it only moves the boundary shared by two EXISTING
    entries — so a plan that was coherent before the move stays coherent
    after it (see docs/run-local-segment-range-regeneration.md's stitch-plan
    audit). The one shape this rejects is an entry this module cannot
    resolve at all, which would otherwise surface as a confusing failure only
    after credits are spent.
    """
    if not stitch_plan:
        return
    for pos, entry in enumerate(stitch_plan):
        kind = entry.get("kind") if isinstance(entry, dict) else None
        if kind not in ("segment", "library"):
            raise RangeValidationError(
                f"stitch plan entry {pos} has an unrecognised shape — cannot "
                "safely re-stitch a candidate for this run"
            )


# ---------------------------------------------------------------------------
# Attempt creation (DB write, no I/O beyond saving already-uploaded override
# files — called synchronously from the API request handler)
# ---------------------------------------------------------------------------


def create_range_attempt(
    *,
    run: Run,
    rs: RunSegment,
    project: VideoProject,
    start_sec: float,
    end_sec: float,
    prompt_override: Optional[str],
    reference_image_urls_override: Optional[list],
    attempt_id: str,
    db: Session,
) -> RunSegmentRangeAttempt:
    """Validate and build (add + flush, but do NOT commit or enqueue) a new
    RunSegmentRangeAttempt row.

    *attempt_id* is generated by the caller (mirrors api_v2._build_copied_run)
    so any reference-override files can be saved into this attempt's own
    directory before the row exists.
    """
    base_snapshot = list(run.segment_snapshot or [])
    if not base_snapshot:
        raise RangeValidationError(
            "this run has no run-local timeline (it predates run-local "
            "segment snapshots) — range editing is not available for it"
        )
    duration = project.duration_sec
    if duration is None:
        raise RangeValidationError("source duration is not known for this project")

    candidate_snapshot, moved_prev_id, moved_next_id = compute_candidate_snapshot(
        base_snapshot, rs.segment_def_id, start_sec, end_sec, duration,
    )
    validate_model_duration(run.model, start_sec, end_sec)
    validate_stitch_plan_coherence(run.stitch_plan)

    attempt = RunSegmentRangeAttempt(
        id=attempt_id,
        run_id=run.id,
        run_segment_id=rs.id,
        requested_start_sec=start_sec,
        requested_end_sec=end_sec,
        prompt_override=prompt_override,
        reference_image_urls_override=reference_image_urls_override,
        base_segment_snapshot=base_snapshot,
        candidate_segment_snapshot=candidate_snapshot,
        keep_recut_paths={
            sd_id: None for sd_id in (moved_prev_id, moved_next_id) if sd_id
        } or None,
        status=RangeAttemptStatus.queued,
    )
    db.add(attempt)
    db.flush()
    return attempt


# ---------------------------------------------------------------------------
# Submit phase
# ---------------------------------------------------------------------------


def submit_range_attempt(
    attempt_id: str, *, kie: Optional[KieClient] = None,
) -> Optional[str]:
    """Cut the candidate swap clip from the ORIGINAL source, upload it, and
    create (or resume) this attempt's kie.ai video task.

    Resume-aware exactly like ``pipeline_segment_image.submit_segment_image_job``:
    if ``attempt.seedance_task_id`` is already set, the existing task is
    returned for polling rather than recreated — a retry (worker restart, or
    a duplicate enqueue) must never rebill kie.ai for the same attempt.

    Returns the kie.ai task id to poll, or ``None`` when the attempt is
    already terminal.
    """
    if kie is None:
        kie = get_shared_client()

    with get_session() as session:
        attempt: Optional[RunSegmentRangeAttempt] = session.get(
            RunSegmentRangeAttempt, attempt_id
        )
        if attempt is None:
            raise ValueError(f"RunSegmentRangeAttempt not found: {attempt_id}")

        if attempt.status in (RangeAttemptStatus.done, RangeAttemptStatus.failed):
            log.info(
                "submit_range_attempt: attempt %s already terminal (status=%s) "
                "— nothing to do", attempt_id, attempt.status,
            )
            return None

        if attempt.seedance_task_id:
            log.info(
                "submit_range_attempt: attempt %s has existing task_id=%s — "
                "resuming without recreating", attempt_id, attempt.seedance_task_id,
            )
            if attempt.status == RangeAttemptStatus.queued:
                transition(attempt, RangeAttemptStatus.generating)
                # Release the submit-phase claim (if any) now that the phase
                # it guarded is over — the finalize phase reuses these same
                # two columns and must see this row as unclaimed, not wait
                # out a stale lease.
                attempt.claim_owner = None
                attempt.claim_expires_at = None
                try:
                    session.commit()
                except StaleDataError:
                    # Another session already advanced this same resume — its
                    # outcome is authoritative; the task id we read is correct
                    # either way (it never changes once set), so just report it.
                    session.rollback()
                    log.info(
                        "submit_range_attempt: attempt %s lost a status-"
                        "transition race while resuming — reporting the "
                        "already-persisted task id anyway", attempt_id,
                    )
            return attempt.seedance_task_id

        try:
            rs: Optional[RunSegment] = session.get(RunSegment, attempt.run_segment_id)
            run: Optional[Run] = session.get(Run, attempt.run_id)
            if rs is None or run is None:
                raise ValueError(
                    f"submit_range_attempt: RunSegment/Run missing for attempt {attempt_id}"
                )
            project: Optional[VideoProject] = session.get(VideoProject, run.project_id)
            if project is None:
                raise ValueError(
                    f"submit_range_attempt: project missing for run {run.id}"
                )
            source = project.source_local_path
            if not source or not os.path.exists(source):
                raise FileNotFoundError(
                    f"Source file not found at {source!r} for project {run.project_id}"
                )

            # ---- Claim the submit phase before ANY side effect (cutting,
            # uploading, resolving reference urls, or calling createTask) ----
            now = datetime.now(timezone.utc)
            if _is_claim_live(attempt, now):
                log.info(
                    "submit_range_attempt: attempt %s is already claimed by "
                    "another worker (until %s) — exiting benignly",
                    attempt_id, attempt.claim_expires_at,
                )
                return None
            if attempt.submit_billed_at is not None:
                # The previous claim (now presumed dead — we already know it
                # is not LIVE, from the check above) got far enough to commit
                # submit_billed_at but never persisted a seedance_task_id.
                # kie.ai's createTask has no idempotency key or lookup-by-
                # reference (kie_client.KieClient), so whether it actually
                # created a billed task cannot be determined here. Do not
                # retry — surface it for manual reconciliation instead (see
                # models.RunSegmentRangeAttempt.submit_billed_at).
                error_msg = _ambiguous_submit_error(attempt.submit_billed_at)
                _fail_attempt(session, attempt, error_msg)
                kcap.range_ambiguous(attempt_id, attempt.submit_billed_at)  # AICP-299 A3; never raises
                log.error(
                    "submit_range_attempt: attempt %s has an AMBIGUOUS prior "
                    "submission (submit_billed_at=%s, no seedance_task_id) — "
                    "marked failed for manual reconciliation, NOT retried",
                    attempt_id, attempt.submit_billed_at,
                )
                return None
            claim_token = uuid.uuid4().hex
            attempt.claim_owner = claim_token
            attempt.claim_expires_at = now + timedelta(seconds=_CLAIM_LEASE_SEC)
            try:
                session.commit()
            except StaleDataError:
                session.rollback()
                log.info(
                    "submit_range_attempt: attempt %s lost the submit-claim "
                    "race to another worker — exiting benignly", attempt_id,
                )
                return None

            attempt_dir = range_attempt_dir(attempt.id, rs.id, run.id, run.project_id)
            claim_dir = _claim_dir(attempt_dir, claim_token)

            # Effective reference set — same three-way priority as the main
            # submit path (pipeline_v2._build_submit_state): this attempt's
            # OWN override (submitted atomically with the range) takes the
            # place of RunSegment.reference_image_urls_override; if this
            # attempt carries none, fall back to the segment's EXISTING
            # generated_reference_urls / reference_image_urls_override /
            # run-level set, exactly what a plain re-run would use.
            if attempt.reference_image_urls_override:
                ref_source = list(attempt.reference_image_urls_override)
            elif rs.generated_reference_urls:
                ref_source = list(rs.generated_reference_urls)
            elif rs.reference_image_urls_override:
                ref_source = list(rs.reference_image_urls_override)
            else:
                ref_source = list(run.reference_image_urls or [])
            ref_urls = resolve_reference_urls(ref_source, kie, gdrive=None)

            voice_ref_urls: list[str] = []
            generate_audio = generate_audio_for_run(run)
            if (
                run.reference_audio_urls
                and ai_models.supports_reference_audio(run.model)
                and generate_audio
            ):
                voice_ref_urls = resolve_reference_urls(
                    list(run.reference_audio_urls), kie, gdrive=None,
                )

            from .pipeline_v2 import effective_mute_source

            mute_audio = effective_mute_source(run, project) or ai_models.is_omni(run.model)
            include_audio = not ai_models.is_omni(run.model) and not mute_audio

            # Claim-scoped: no two claims of this attempt (concurrent racers,
            # or a later takeover) ever write this path.
            clip_dst = os.path.join(claim_dir, "clip.mp4")
            media_mod.cut_clip(
                source, attempt.requested_start_sec, attempt.requested_end_sec,
                clip_dst, include_audio=include_audio,
            )
            attempt.local_clip_path = clip_dst
            try:
                session.commit()
            except StaleDataError:
                session.rollback()
                _cleanup_claim(claim_dir)
                log.warning(
                    "submit_range_attempt: attempt %s lost its claim while "
                    "cutting the candidate clip — discarding this worker's "
                    "cut and exiting benignly", attempt_id,
                )
                return None

            # Namespaced by attempt id + claim token (both globally unique) —
            # same collision reasoning as the main pipeline's rs_id-namespaced
            # uploads, plus defense-in-depth against a takeover reusing the
            # same remote object name.
            clip_url = kie.upload_file(
                clip_dst, "charswap/segments",
                file_name=f"{attempt.id}_{claim_token}_{os.path.basename(clip_dst)}",
            )

            # Same three-way priority as the reference set just above: this
            # attempt's OWN prompt override (submitted atomically with the
            # range) first; if this attempt carries none, fall back to the
            # segment's EXISTING hand override rather than skipping straight
            # to the run-level prompt — exactly what a plain re-run would use,
            # and what the product contract calls "per-segment prompt
            # overrides remain applicable" for a range-only regeneration.
            effective_prompt = (
                attempt.prompt_override or rs.prompt_override or (run.prompt or "")
            )

            # Knowledge capture A3 (AICP-299): the intent, BEFORE the
            # submit_billed_at commit — an interrupt here leaves only the claim,
            # exactly like a crash during the upload. Never raises.
            kb_recipe = dict(
                run_model=run.model,
                effective_prompt=effective_prompt,
                ref_urls=ref_urls,
                clip_url=clip_url,
                clip_start=attempt.requested_start_sec,
                clip_end=attempt.requested_end_sec,
                run_resolution=run.resolution,
                project_aspect_ratio=project.aspect_ratio,
                project_width=project.width,
                project_height=project.height,
                generate_audio=generate_audio,
                audio_ref_urls=voice_ref_urls,
            )
            kb_attempt = kcap.swap_intent(
                rs_id=rs.id, recipe=kb_recipe, include_audio=include_audio, attempt_kind="range_attempt",
                origin="operator", target_kind="range_attempt", target_id=attempt.id,
                idempotency_key=f"range_attempt:{attempt.id}",
                ref_override=attempt.reference_image_urls_override, site="A3",
            )

            # Mark the ambiguous-billing window BEFORE the one paid call this
            # module makes — see submit_billed_at's docstring in models.py.
            # Committed (and version-checked) separately from the createTask
            # call itself so a lost claim race here is caught BEFORE anything
            # is spent, never confused with the genuinely ambiguous window
            # that starts only once this commit has actually landed.
            attempt.submit_billed_at = datetime.now(timezone.utc)
            try:
                session.commit()
            except StaleDataError:
                session.rollback()
                _cleanup_claim(claim_dir)
                log.info(
                    "submit_range_attempt: attempt %s lost its claim just "
                    "before the billed call — nothing was spent by this "
                    "worker; exiting benignly", attempt_id,
                )
                return None

            # Re-verify ownership immediately before the one paid call this
            # module makes — defense in depth on top of the job-timeout-
            # sized lease (see _assert_still_owner's docstring). A takeover
            # this narrow window would still be caught by the version-
            # checked commit right after the call, but checking BEFORE it
            # means a stale owner never makes the billed call at all,
            # rather than making it and then failing to record it.
            try:
                _assert_still_owner(session, attempt_id, claim_token)
            except _ClaimLost:
                _cleanup_claim(claim_dir)
                log.info(
                    "submit_range_attempt: attempt %s lost its claim just "
                    "before the billed call (caught by an explicit re-check) "
                    "— nothing was spent by this worker; exiting benignly",
                    attempt_id,
                )
                return None

            try:
                task_id = _create_swap_task(kie=kie, **kb_recipe)
            except Exception as exc:
                kcap.dispatch_error(kb_attempt, exc, site="A3")
                raise
            # Measured the moment kie.ai accepted it — before the commit
            # below, which can lose the claim but not undo the bill.
            spend_mod.record_draw(
                spend_mod.range_scope(attempt_id),
                spend_mod.video_credits(
                    run.model, run.resolution,
                    attempt.requested_end_sec - attempt.requested_start_sec,
                ),
                kind="range video", call_key=task_id, sessions=get_session,
            )
            attempt.seedance_task_id = task_id
            transition(attempt, RangeAttemptStatus.generating)
            # Release the submit-phase claim now that the phase it guarded
            # is over — the finalize phase reuses these same two columns and
            # must see this row as unclaimed, not wait out a stale lease.
            attempt.claim_owner = None
            attempt.claim_expires_at = None
            try:
                session.commit()
            except StaleDataError:
                # We DID make the paid createTask call and have a real,
                # valid task_id in hand — but lost the claim (our own lease
                # expired and something else took over) before we could
                # persist it. This is the same ambiguity submit_billed_at
                # exists to flag, seen from the side that actually knows the
                # call succeeded; we must not force this task_id onto a row
                # a takeover may have already marked failed/ambiguous, and we
                # must not silently drop it either — log loudly with the
                # real task_id for manual reconciliation, and leave whatever
                # the other session decided in place. A generous
                # _CLAIM_LEASE_SEC (comfortably longer than a normal
                # cut+upload+createTask round) makes this vanishingly rare;
                # see docs/run-local-segment-range-regeneration.md.
                session.rollback()
                log.error(
                    "submit_range_attempt: attempt %s created kie.ai task "
                    "%s but LOST its claim before persisting the task id — "
                    "this task may be orphaned server-side; check the "
                    "kie.ai dashboard for task %s and reconcile manually",
                    attempt_id, task_id, task_id,
                )
                return None
            kcap.submitted(kb_attempt, task_id, site="A3")  # after the commit (P1)
            log.info(
                "submit_range_attempt: attempt %s (segment %s, run %s) → "
                "task %s range=[%.2f, %.2f)",
                attempt_id, attempt.run_segment_id, attempt.run_id, task_id,
                attempt.requested_start_sec, attempt.requested_end_sec,
            )
            return task_id
        except _ClaimLost:
            # Every call site above that can raise this already handles it
            # locally with its own benign `return None` — this is a
            # defensive backstop only, so a FUTURE _assert_still_owner call
            # added without its own local handling never falls through to
            # the generic failure path below and wrongly fails a benignly-
            # superseded attempt.
            log.info(
                "submit_range_attempt: attempt %s lost ownership (caught by "
                "the outer backstop, not a local check) — exiting benignly",
                attempt_id,
            )
            return None
        except Exception as exc:
            log.exception("submit_range_attempt failed: attempt_id=%s", attempt_id)
            if attempt.submit_billed_at is not None and not attempt.seedance_task_id:
                # This exception fired at or after the commit that precedes
                # kie.ai's paid createTask call — even a client-side
                # exception (a timeout, or tenacity exhausting its retries)
                # does NOT prove the request never reached kie.ai's server;
                # it only proves WE never got a confirmed response. Flag it
                # with the same ambiguous-billing message an expired-claim
                # takeover would use, folding in the real exception for
                # diagnostics, instead of recording only the raw error text
                # (found by the mandatory Cursor CLI review gate — a plain
                # "ReadTimeout" message reads as a clean failure and invites
                # exactly the retry this whole mechanism exists to prevent).
                attempt.error_message = (
                    f"{_ambiguous_submit_error(attempt.submit_billed_at)} "
                    f"(underlying error: {_truncate_error(exc)})"
                )
            else:
                attempt.error_message = _truncate_error(exc)
            if attempt.status not in (RangeAttemptStatus.done, RangeAttemptStatus.failed):
                try:
                    transition(attempt, RangeAttemptStatus.failed)
                except Exception:
                    attempt.status = RangeAttemptStatus.failed
            try:
                session.commit()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# Poll phase — on success, also stitches, delivers and PROMOTES.
# ---------------------------------------------------------------------------


def _fail_attempt(session, attempt: RunSegmentRangeAttempt, message: str) -> None:
    attempt.error_message = message
    if attempt.status not in (RangeAttemptStatus.done, RangeAttemptStatus.failed):
        try:
            transition(attempt, RangeAttemptStatus.failed)
        except Exception:
            attempt.status = RangeAttemptStatus.failed
    session.commit()


def _resolve_keep_clip(
    sd: dict, source: str, run_id: str, project_id: str,
    *,
    session: Optional[Session] = None,
    attempt_id: Optional[str] = None,
    claim_token: Optional[str] = None,
    claim_dir: Optional[str] = None,
) -> str:
    """The clip for an UNAFFECTED `keep` timeline entry (same bounds as the
    run's active snapshot) — reuse the run's own cached keep clip (cut once
    by the original stitch), cutting it if it does not exist yet.

    Path is keyed by INDEX and BOUNDS (storage.keep_clip_path) — see that
    function's docstring; this must stay in sync with
    pipeline_v2._cut_source, which caches the SAME clips for the main
    pipeline's own stitch.

    This cache path is SHARED across every caller (every attempt on this
    run, plus the main pipeline) — never claim-scoped by design, since it
    is meant to be reused. When *claim_token*/*claim_dir* are given (the
    normal case — see _stitch_candidate), a missing clip is cut into a
    claim-scoped TEMP path first, ownership is re-verified
    (_assert_still_owner), and only then is it published into the shared
    *dst* via ``os.replace`` — atomic on the same filesystem, so a
    concurrent reader's ``os.path.exists``/``getsize`` check here always
    sees either nothing yet or the FULLY-written file, never a partial one,
    and a stale/superseded caller's cut (caught by the ownership check)
    never overwrites a newer publish. Without this, two callers finding the
    same missing entry could both run ffmpeg with ``-y`` straight into
    *dst* at once (found by an independent audit — the module docstring's
    "every worker writes only inside claims/<token>/" claim did not hold
    for this one path).

    *claim_token*/*claim_dir* default to ``None`` for a caller with no
    claim to scope to (a direct unit-test invocation, or a future caller
    outside the claimed pipeline) — that case falls back to the original
    direct-cut behaviour, unchanged.
    """
    dst = keep_clip_path(run_id, project_id, sd["index"], sd["start_sec"], sd["end_sec"])
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    if claim_token is None or claim_dir is None:
        media_mod.cut_clip(source, sd["start_sec"], sd["end_sec"], dst, fast_intermediate=True)
        return dst
    tmp = _staging_path(claim_dir, dst)
    media_mod.cut_clip(source, sd["start_sec"], sd["end_sec"], tmp, fast_intermediate=True)
    if session is not None and attempt_id is not None:
        _assert_still_owner(session, attempt_id, claim_token)
    os.replace(tmp, dst)
    return dst


def _stitch_candidate(
    session, run: Run, project: VideoProject, attempt: RunSegmentRangeAttempt,
    rs_map: dict, work_dir: str, claim_token: Optional[str] = None,
) -> None:
    """Build the candidate's own final video (or leave it unset for a
    no-stitch run) from ``attempt.candidate_segment_snapshot``.

    Every entry NOT touched by this attempt resolves to the run's EXISTING
    result/clip — no unaffected swap segment is resubmitted or re-billed, and
    no unaffected keep clip is re-cut. Mirrors
    ``pipeline_v2._run_stitch_and_deliver``'s clip resolution, scoped to a
    single-segment candidate and writing into this attempt's own paths only.

    *work_dir* is the CALLING claim's own scoped directory
    (``_claim_dir(attempt_dir, claim_token)``), never the bare attempt
    directory — two claims of the same attempt (concurrent racers, or a
    later takeover) must never write the same muxed/final path.

    *claim_token* (``None`` for a direct unit-test call with no claim to
    scope to) is forwarded to ``_resolve_keep_clip`` for the ONE path this
    function writes that is NOT under *work_dir* — the shared, bounds-keyed
    keep-clip cache — so a missing entry is staged in *work_dir* and
    published atomically, ownership-checked, rather than cut in place.
    """
    source = project.source_local_path
    info = media_mod.probe(source)
    width, height, fps = media_mod.get_default_target(info)
    attempt_dir = work_dir
    candidate_snapshot = attempt.candidate_segment_snapshot
    moved_ids = set((attempt.keep_recut_paths or {}).keys())

    plan: Optional[list] = list(run.stitch_plan) if run.stitch_plan is not None else None
    if plan is not None and not plan:
        # No-stitch run: the candidate result clip IS the deliverable: no
        # final video to build.
        return

    audio_mode = run.audio_mode or "original"
    if not ai_models.spec_for(run.model).produces_audio:
        audio_mode = "original"

    if plan is None:
        entries = [{"kind": "segment", "segment_def_id": e["id"]} for e in candidate_snapshot]
        stitch_audio_mode = audio_mode
    else:
        entries = plan
        stitch_audio_mode = "seedance"

    sd_by_id = {e["id"]: e for e in candidate_snapshot}
    lib_paths: dict[str, str] = {}
    for pos, entry in enumerate(entries):
        kind = entry.get("kind") if isinstance(entry, dict) else None
        if kind == "segment":
            if entry.get("segment_def_id") not in sd_by_id:
                raise RuntimeError(
                    f"stitch plan entry {pos}: segment_def_id "
                    f"{entry.get('segment_def_id')!r} not in candidate timeline"
                )
        elif kind == "library":
            lib = session.get(LibraryClip, entry.get("clip_id"))
            if lib is None or not lib.local_path or not os.path.exists(lib.local_path):
                raise RuntimeError(
                    f"stitch plan entry {pos}: library clip file is missing"
                )
            lib_paths[entry["clip_id"]] = lib.local_path
        else:
            raise RuntimeError(f"stitch plan entry {pos}: unknown kind {kind!r}")

    def _clip_for_entry(entry: dict) -> str:
        if entry["kind"] == "library":
            return lib_paths[entry["clip_id"]]
        sd = sd_by_id[entry["segment_def_id"]]
        if sd["action"] == "swap" and sd["id"] == rs_map["target_segment_def_id"]:
            clip = attempt.local_result_path
        elif sd["id"] in moved_ids:
            clip = attempt.keep_recut_paths[sd["id"]]
        elif sd["action"] == "keep":
            clip = _resolve_keep_clip(
                sd, source, run.id, run.project_id,
                session=session, attempt_id=attempt.id,
                claim_token=claim_token, claim_dir=work_dir,
            )
        else:
            existing_rs = rs_map["by_segment_def_id"].get(sd["id"])
            if (
                existing_rs is None
                or not existing_rs.local_result_path
                or not os.path.exists(existing_rs.local_result_path)
            ):
                raise RuntimeError(
                    f"segment {sd['index']} is not completed — cannot build "
                    "a candidate final around it"
                )
            clip = existing_rs.local_result_path

        if plan is not None and sd["action"] == "swap" and audio_mode == "original":
            muxed = os.path.join(attempt_dir, f"resultaud_{sd['index']:04d}.mp4")
            if os.path.exists(muxed) and os.path.getmtime(muxed) >= os.path.getmtime(clip):
                return muxed
            media_mod.mux_original_audio(
                clip, source, sd["start_sec"], muxed, source_has_audio=info.has_audio,
            )
            return muxed
        return clip

    clip_paths = [_clip_for_entry(e) for e in entries]
    final_dst = os.path.join(attempt_dir, "final.mp4")
    log.info(
        "attempt %s: stitching %d clips → %s plan=%s audio=%s",
        attempt.id, len(clip_paths), final_dst,
        "custom" if plan is not None else "default", stitch_audio_mode,
    )
    # The same conform as the run's own stitch (pipeline_v2.stitch_frame_counts):
    # a promoted candidate must not bring the clip-length drift back.
    from types import SimpleNamespace as _NS
    from .pipeline_v2 import stitch_frame_counts

    frame_counts = stitch_frame_counts(
        [_NS(**e) for e in candidate_snapshot], fps, plan=plan, audio_mode=stitch_audio_mode,
    )
    media_mod.stitch(
        clip_paths, audio_source=source, dst=final_dst,
        width=width, height=height, fps=fps, audio_mode=stitch_audio_mode,
        frame_counts=frame_counts,
    )
    try:
        media_mod.enforce_size_limit(final_dst)
    except Exception as exc:
        log.warning(
            "attempt %s: could not shrink %s to the delivery size cap: %s",
            attempt.id, final_dst, exc,
        )
    attempt.final_local_path = final_dst
    try:
        attempt.result_duration_sec = media_mod.probe(final_dst).duration_sec
    except Exception as exc:
        log.warning("attempt %s: failed to probe %s: %s", attempt.id, final_dst, exc)
    attempt.source_fallback_segments = run.source_fallback_segments or 0


def _pin_candidate_final(final_path: str) -> str:
    """A private hard link of the candidate final, for the in-place upload.

    The AICP-342 "private delivery link": the upload reads a name nothing else
    writes, and :func:`_promote` checks it is still the same inode as
    ``attempt.final_local_path`` right before the transfer — so the bytes put
    behind the run's Drive id are exactly the file the run is re-pointed at.
    The candidate final already lives in this claim's own directory, so this
    is defence in depth rather than the only guard (finalize needs it more:
    its ``final.mp4`` is a shared name).
    """
    pin = os.path.join(
        os.path.dirname(final_path), f".deliver-{uuid.uuid4().hex}.mp4"
    )
    os.link(final_path, pin)
    return pin


def _promote(
    session, run: Run, rs: RunSegment, attempt: RunSegmentRangeAttempt,
    claim_token: Optional[str] = None, claim_dir: Optional[str] = None,
    gdrive: Optional[GDriveClient] = None,
) -> None:
    """Copy the successful candidate's fields onto the real Run/RunSegment —
    one coherent version, in the same commit that marks the attempt `done`.

    **Delivery in place (AICP-349).** When the run already has a Drive file
    the candidate final replaces that file's CONTENT (``files().update``) —
    the same id, so the Creative Link, the Job ID and an already-exported
    Final Creatives row keep serving the run, exactly like a segment rerun
    (AICP-342) and an audio patch. It happens here, INSIDE the Run-row lock
    and after every staleness check, as the last fallible step before the
    commit (the pipeline_audio_patch._promote order): a candidate that loses
    the race or fails a check never touches Drive, and a failed update raises
    before anything is committed, so the run stays byte-for-byte as it was.
    The claim is re-checked under a row lock on the ATTEMPT too (the
    range-attempt equivalent of finalize's processing-lock ownership): a
    takeover's claim commit blocks behind this transaction instead of
    slipping in between the check and the upload. No fallback to a fresh
    upload — a deleted Drive file fails the attempt (AICP-348 is the explicit
    re-link). Only a run with NO Drive file yet points at the fresh upload
    the deliver step made (``attempt.gdrive_file_id``).

    Drive ahead of the row is closed where it can be: an update that reports
    failure is settled by Drive's md5 (landed -> promote; not landed or a 404
    -> fail, Drive untouched; cannot tell -> fail with a message saying the
    link MAY hold the candidate), and a commit that fails after the update
    landed hands the attempt back for a fresh finalize
    (:class:`_PromoteUncommitted`) instead of failing it. Residual, the one
    audio patch and finalize accept: a worker KILLED between Drive
    acknowledging and the commit; the reaper resumes the attempt (its task
    id is persisted), which re-sends an equivalent final.

    *claim_token*/*claim_dir* (``None`` for the existing direct unit-test
    invocations that exercise the Run-row-lock behaviour in isolation) gate
    the keep-clip cache warm-up below: when given, that copy is staged in
    *claim_dir* and published atomically, ownership-re-checked, exactly
    like ``_resolve_keep_clip`` — the SAME shared cache path this
    function's own warm-up writes to.

    Only ever called for a run that was (and — enforced defensively here —
    still is) ``done``: range regeneration is deliberately gated to `done`
    runs only (api_v2's ``_RANGE_REGEN_ELIGIBLE_RUN_STATUSES`), which sidesteps
    a much harder question this feature does not need to answer — whether
    fixing ONE segment should also flip a `failed`/`incomplete` run (with
    OTHER still-broken segments) to `done`. Since ``run.status`` never leaves
    `done` for the whole lifetime of an in-flight attempt (nothing in this
    module touches it before this point), promotion never needs to — and
    must not — transition it.

    Locks the Run row for the rest of this transaction before checking or
    touching anything on it. The DB-level partial unique index makes it
    impossible for a SECOND attempt to be active on this run at the same
    time, but it does not stop the SAME attempt being processed by two
    worker sessions at once (a duplicate poll enqueue, or a recovery
    re-enqueue racing a still-running job — both have happened for other
    derived creatives in this codebase). Without the lock, two sessions can
    each load `run` before either commits, each pass the staleness check
    below against their own stale in-memory copy, and then both commit —
    last write wins, silently discarding whichever promotion committed
    first. session.refresh(..., with_for_update=True) re-reads the row
    under a real row lock and overwrites this object's attributes with
    whatever is CURRENTLY persisted, so the loser's staleness check below
    sees the winner's already-promoted snapshot and correctly refuses. A
    no-op on SQLite (no row-level locking there); the concurrency guarantee
    is proved against PostgreSQL, this feature's production database.
    """
    if claim_token is not None:
        _assert_still_owner(session, attempt.id, claim_token)
    session.refresh(run, with_for_update=True)
    if claim_token is not None:
        # Lock order: Run row, then attempt row (nothing takes them the other
        # way round). The attempt carries no uncommitted change here — every
        # step before _promote commits — so the refresh discards nothing.
        session.refresh(attempt, with_for_update=True)
        if attempt.claim_owner != claim_token:
            raise _ClaimLost(
                f"attempt {attempt.id}: claim_owner is now "
                f"{attempt.claim_owner!r}, no longer this call's token "
                f"{claim_token!r} — not promoting"
            )
    if run.status != RunStatus.done:
        raise RuntimeError(
            f"refusing to promote: run status changed to {run.status!r} "
            "while this attempt was in flight — the last successful result "
            "is left untouched"
        )
    if run.segment_snapshot != attempt.base_segment_snapshot:
        # A DIFFERENT attempt on this run already promoted while this one was
        # in flight (the API's active-attempt 409 gate is a non-locking
        # SELECT, so two concurrent range-rerun requests can both pass it and
        # both reach here). Refuse rather than silently overwrite the other
        # attempt's already-promoted timeline with one built from a now-STALE
        # base — that would desync run.segment_snapshot from the OTHER
        # attempt's already-promoted rs.local_result_path.
        raise RuntimeError(
            "refusing to promote: the run's active timeline changed since "
            "this attempt was created (another attempt already promoted) — "
            "this candidate was NOT applied; review the current timeline "
            "and start a new attempt if a correction is still needed"
        )

    run.segment_snapshot = copy.deepcopy(attempt.candidate_segment_snapshot)

    # Warm the run's clip cache for whichever `keep` neighbour(s) this
    # attempt moved, at the NEW bounds' key (storage.keep_clip_path) — pure
    # optimisation, not a correctness requirement: _resolve_keep_clip (this
    # module) and pipeline_v2._cut_source both already key their reuse check
    # off (index, bounds), so a later reader asking for the just-promoted
    # bounds would re-cut from source and land here anyway if this copy is
    # skipped or fails. Crucially this WRITES ONLY the new-bounds path, never
    # the old one — so even a crash or exception between this copy and the
    # commit below cannot corrupt the file a still-active OLD snapshot would
    # read (that old-bounds file, if it exists, is never touched by a
    # promotion at all). See keep_clip_path's docstring for the full
    # invariant this replaces (the old index-only cache path was NOT
    # crash-safe: a copy that lands but whose commit fails left the OLD
    # bounds' filename holding NEW bytes).
    candidate_by_id = {e["id"]: e for e in attempt.candidate_segment_snapshot}
    for sd_id, recut_path in (attempt.keep_recut_paths or {}).items():
        if not recut_path or not os.path.exists(recut_path):
            continue
        entry = candidate_by_id.get(sd_id)
        if entry is None:
            continue
        cache_dst = keep_clip_path(
            run.id, run.project_id, entry["index"], entry["start_sec"], entry["end_sec"],
        )
        try:
            if claim_token is not None and claim_dir is not None:
                tmp = _staging_path(claim_dir, cache_dst)
                shutil.copy2(recut_path, tmp)
                _assert_still_owner(session, attempt.id, claim_token)
                os.replace(tmp, cache_dst)
            else:
                shutil.copy2(recut_path, cache_dst)
        except _ClaimLost:
            # Must NOT be swallowed as a mere cache-warm hiccup — this
            # worker's ownership was superseded, so the WHOLE promotion
            # must abort, not just this optimisation. Propagate to the
            # caller's except (StaleDataError, _ClaimLost) handling.
            raise
        except Exception as exc:
            log.warning(
                "attempt %s: could not warm keep clip cache %s: %s — a "
                "later reader will just re-cut it from source, correctly",
                attempt.id, cache_dst, exc,
            )

    if attempt.prompt_override is not None:
        rs.prompt_override = attempt.prompt_override or None
    if attempt.reference_image_urls_override is not None:
        rs.reference_image_urls_override = list(attempt.reference_image_urls_override)
    rs.status = SegmentStatus.completed
    rs.error_message = None
    rs.source_fallback = False
    rs.seedance_task_id = attempt.seedance_task_id
    rs.seedance_result_url = attempt.seedance_result_url
    rs.cost_credits = attempt.cost_credits
    rs.local_clip_path = attempt.local_clip_path
    rs.local_result_path = attempt.local_result_path

    # Read under the Run-row lock: the Drive id the sheet holds RIGHT NOW.
    in_place_file_id: Optional[str] = None
    if attempt.final_local_path:
        run.result_local_path = attempt.final_local_path
        run.result_duration_sec = attempt.result_duration_sec
        run.source_fallback_segments = attempt.source_fallback_segments
        if run.result_gdrive_file_id:
            # AICP-349: replace the content behind the existing id (below).
            # A fresh upload this attempt may have made (the run had no
            # Drive file when the deliver step looked) is left unreferenced
            # rather than moving a link that now exists.
            in_place_file_id = run.result_gdrive_file_id
        elif attempt.gdrive_file_id:
            run.result_gdrive_file_id = attempt.gdrive_file_id
    run.error_message = None

    attempt.promoted_at = datetime.now(timezone.utc)
    attempt.error_message = None
    transition(attempt, RangeAttemptStatus.done)

    if in_place_file_id is not None:
        # Last fallible step before the commit — see the docstring.
        pin = _pin_candidate_final(attempt.final_local_path)
        try:
            try:
                same = os.path.samefile(pin, attempt.final_local_path)
            except OSError:
                same = False
            if not same:
                raise RuntimeError(
                    "refusing in-place delivery: the candidate final was "
                    "replaced while this attempt was promoting — Drive left "
                    "untouched"
                )
            if gdrive is None:
                gdrive = GDriveClient()
            try:
                _deliver_in_place_with_retry(gdrive, in_place_file_id, pin, run.id)
            except Exception as exc:
                # The transfer may have landed with its response lost: settle
                # it by content before calling it a failure, or a "failed"
                # attempt could leave the exported link serving its bytes.
                landed = _drive_holds(gdrive, in_place_file_id, pin)
                if landed is None:
                    raise RuntimeError(
                        f"in-place Drive delivery to the run's file {in_place_file_id} "
                        f"failed and its outcome could not be verified: {exc}. The run "
                        "was NOT promoted, but the Drive file MAY already hold this "
                        "candidate — open the run's Drive link to check; a new range "
                        "attempt or a segment rerun re-delivers."
                    ) from exc
                if not landed:
                    raise RuntimeError(
                        f"in-place Drive delivery to the run's file {in_place_file_id} "
                        f"failed: {exc}. The run keeps its previous result and link "
                        "(no fresh upload: that would change the Creative Link). If "
                        "the Drive file was deleted, the run has to be re-linked to "
                        "a new Drive file before a range attempt can deliver."
                    ) from exc
                log.warning(
                    "attempt %s: Drive update of %s reported %s, but the file "
                    "already holds this candidate (md5 match) — promoting",
                    attempt.id, in_place_file_id, exc,
                )
        finally:
            try:
                os.remove(pin)
            except OSError:
                pass
        attempt.gdrive_file_id = in_place_file_id

    try:
        session.commit()
    except StaleDataError:
        raise
    except Exception as exc:
        if in_place_file_id is None:
            raise
        # Drive already holds the candidate; failing the attempt now would
        # leave the exported link ahead of a run that still describes the
        # old result. Hand it back for a fresh finalize instead (see
        # _PromoteUncommitted).
        raise _PromoteUncommitted(
            f"Drive file {in_place_file_id} holds this candidate but the "
            f"promotion did not commit: {exc}"
        ) from exc
    log.info(
        "attempt %s: PROMOTED onto run %s segment %s — new range=[%.2f, %.2f)%s",
        attempt.id, run.id, rs.id,
        attempt.requested_start_sec, attempt.requested_end_sec,
        f", Drive file {in_place_file_id} replaced in place" if in_place_file_id else "",
    )


class _PromoteUncommitted(Exception):
    """The in-place Drive update landed but the promotion commit did not.

    Not a failure of the candidate: the caller puts the attempt back to
    ``generating`` with its claim released (the reaper's resume shape), so
    the next poll redoes download -> stitch -> promote and re-sends the same
    content to the same id, which converges Drive and the row again.
    """


def _md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _drive_holds(gdrive, file_id: str, path: str) -> Optional[bool]:
    """Does Drive file *file_id* hold exactly *path*'s bytes?

    True / False when Drive answered (a 404 is False: nothing of ours is
    there), None when it could not be told (metadata read failed, no md5).
    """
    reader = getattr(gdrive, "file_md5", None)
    if reader is None:
        return None
    try:
        remote = reader(file_id)
    except GDriveFileNotFound:
        return False
    except Exception as exc:
        log.warning("could not read Drive md5 of %s: %s", file_id, exc)
        return None
    if not remote:
        return None
    try:
        return remote == _md5_of(path)
    except OSError as exc:
        log.warning("could not hash %s: %s", path, exc)
        return None


def _requeue_uncommitted_promotion(
    session, attempt_id: str, claim_token: str, note: str,
) -> None:
    """After :class:`_PromoteUncommitted`: back to ``generating``, claim
    released, so the next poll takes a fresh claim and finalizes again.
    Forced status, the same precedent recovery's resume uses (there is no
    ``delivering -> generating`` edge). Raises if the DB is still failing —
    the job then errors and the reaper resumes the attempt (its task id is
    persisted)."""
    session.rollback()
    attempt = session.get(RunSegmentRangeAttempt, attempt_id)
    if attempt is None or attempt.status in (RangeAttemptStatus.done, RangeAttemptStatus.failed):
        return
    if attempt.claim_owner != claim_token:
        return
    attempt.status = RangeAttemptStatus.generating
    attempt.claim_owner = None
    attempt.claim_expires_at = None
    attempt.error_message = _truncate_error(RuntimeError(
        f"{note} — finalizing again (same content, same Drive file)"
    ))
    session.commit()


def poll_range_attempt_once_published(
    attempt_id: str, *, kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
) -> tuple[str, Optional[PublishedFinal]]:
    """:func:`poll_range_attempt_once` plus the committed publication
    descriptor when THIS call promoted (AICP-299 §4), for the RQ wrapper's
    knowledge link. ``None`` on every other outcome."""
    box: list = []
    status = poll_range_attempt_once(attempt_id, kie=kie, gdrive=gdrive, _published=box)
    return status, (box[0] if box and status == "done" else None)


def poll_range_attempt_once(
    attempt_id: str, *, kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
    _published: Optional[list] = None,
) -> str:
    """One round of status-check for a single attempt's kie.ai task.

    On success: downloads the result, re-cuts any moved `keep` neighbour(s)
    from the ORIGINAL source, builds a candidate final (or none, for a
    no-stitch run), delivers it, and PROMOTES — all in this call, since a
    single-segment stitch + deliver is bounded work (mirrors
    pipeline_v2.finalize_run being driven synchronously once its own
    completeness gate passes). On any failure at any of those steps the
    attempt is marked `failed` and the Run/RunSegment are left untouched.

    Every commit below writes through ``RunSegmentRangeAttempt.version_id``
    (SQLAlchemy's ``version_id_col`` — see the model). A duplicate poll
    enqueue, or a recovery re-enqueue racing a still-running job, can start
    a SECOND call of this function for the SAME attempt while this one is
    mid-flight; without version checking, that second call's own (older,
    in-memory) status could silently overwrite whatever the first call had
    already committed by the time it gets there — including overwriting an
    already-`done`, already-promoted attempt back to an in-progress status,
    which would then fail ITS OWN staleness check in `_promote` and get
    marked `failed`, discarding the true outcome. Every commit site below
    that could race this way is wrapped so a `StaleDataError` — this
    session lost that race — is treated as "abandon my own work, the OTHER
    session's outcome already won", never as a reason to call
    `_fail_attempt`.

    Returns one of ``"pending"``, ``"done"``, ``"failed"``.
    """
    if kie is None:
        kie = get_shared_client()

    with get_session() as session:
        attempt: Optional[RunSegmentRangeAttempt] = session.get(
            RunSegmentRangeAttempt, attempt_id
        )
        if attempt is None:
            raise ValueError(f"RunSegmentRangeAttempt not found: {attempt_id}")
        if attempt.status in (RangeAttemptStatus.done, RangeAttemptStatus.failed):
            return attempt.status.value
        if not attempt.seedance_task_id:
            raise ValueError(f"attempt {attempt_id} has no seedance_task_id to poll")

        try:
            data = kie.get_task(attempt.seedance_task_id)
        except Exception as exc:
            log.warning(
                "poll_range_attempt_once: attempt %s get_task failed: %s — "
                "will retry next round", attempt_id, exc,
            )
            return "pending"

        state = (data.get("state") or "").lower()
        if state != "success" and state != "fail":
            return "pending"
        # Terminal: kie.ai's reported cost replaces the estimate (AICP-155),
        # whatever the branch below makes of the result.
        spend_mod.settle_draw(
            attempt.seedance_task_id, _parse_cost_credits(data),
            scope_key=spend_mod.range_scope(attempt_id), kind="range video",
            sessions=get_session,
        )

        try:
            return _handle_terminal_kie_state(
                session, attempt, attempt_id, data, state, kie=kie, gdrive=gdrive,
                published=_published,
            )
        except (StaleDataError, _ClaimLost):
            # This session's commit lost a version race (StaleDataError) OR
            # an explicit _assert_still_owner re-check independently caught
            # a takeover (_ClaimLost) — a DIFFERENT session already advanced
            # (or finalized) this same attempt, OR (with the finalize-phase
            # claim above) is genuinely still working it right now: either
            # exception is raised whether the other session is already
            # terminal or merely a few commits ahead of us, so this session
            # cannot assume the row is done just because it lost. Report
            # the row's ACTUAL current status, but only if it is one of
            # this function's own terminal values — a non-terminal status
            # (the other session is still mid-flight:
            # `generating`/`stitching`/`delivering`) must still come back
            # as "pending", never leak past this function's documented
            # pending/done/failed contract.
            session.rollback()
            current = session.get(RunSegmentRangeAttempt, attempt_id)
            log.info(
                "poll_range_attempt_once: attempt %s lost a concurrent-"
                "worker version race — abandoning this call, reporting the "
                "other session's outcome (%s)",
                attempt_id, current.status.value if current else "deleted",
            )
            if current is None:
                return "failed"
            if current.status in (RangeAttemptStatus.done, RangeAttemptStatus.failed):
                return current.status.value
            return "pending"


def _handle_terminal_kie_state(
    session, attempt: RunSegmentRangeAttempt, attempt_id: str, data: dict,
    state: str, *, kie: KieClient, gdrive: Optional[GDriveClient],
    published: Optional[list] = None,
) -> str:
    """The success/fail-state body of poll_range_attempt_once, split out so
    its single caller can wrap it in one `except StaleDataError` instead of
    one per commit site (see that function's docstring)."""
    if state == "fail":
        msg = data.get("failMsg") or data.get("failCode") or "unknown error"
        _fail_attempt(session, attempt, f"video generation failed: {msg}")
        kcap.terminal(task_id=attempt.seedance_task_id, status="failed", error_class=str(msg),
                      payload={"failMsg": str(msg)[:500]}, site="T5")
        return "failed"

    url = _parse_result_url(data)
    if not url:
        _fail_attempt(session, attempt, "task succeeded but returned no result URL")
        kcap.terminal(task_id=attempt.seedance_task_id, status="failed", error_class="no_result_url", site="T5")
        return "failed"

    rs: Optional[RunSegment] = session.get(RunSegment, attempt.run_segment_id)
    run: Optional[Run] = session.get(Run, attempt.run_id)
    if rs is None or run is None:
        _fail_attempt(session, attempt, "RunSegment/Run was deleted while this attempt was running")
        return "failed"
    project: Optional[VideoProject] = session.get(VideoProject, run.project_id)
    if project is None:
        _fail_attempt(session, attempt, "project was deleted while this attempt was running")
        return "failed"

    # ---- Claim the finalize phase before ANY side effect (downloading,
    # re-cutting, stitching, delivering or promoting). Unlike the submit
    # phase, an expired claim here is always safe to take over: nothing past
    # this point makes a paid external call — see the module docstring. ----
    claim_token = _try_claim_phase(session, attempt)
    if claim_token is None:
        log.info(
            "poll_range_attempt_once: attempt %s finalize phase is already "
            "claimed by another worker — exiting benignly, reporting "
            "pending", attempt_id,
        )
        return "pending"

    attempt_dir = range_attempt_dir(attempt.id, rs.id, run.id, run.project_id)
    claim_dir = _claim_dir(attempt_dir, claim_token)
    try:
        # Re-verify ownership immediately before EVERY remaining external/
        # publish side effect (defense in depth on top of the job-timeout-
        # sized lease — see _assert_still_owner's docstring). Raises
        # _ClaimLost, caught by this function's own except clause below.
        _assert_still_owner(session, attempt_id, claim_token)
        # Claim-scoped: no two claims of this attempt (concurrent racers, or
        # a later takeover) ever download into, or read from, this path.
        result_dst = os.path.join(claim_dir, "result.mp4")
        try:
            kie.download_result(url, result_dst)
        except Exception as exc:
            _fail_attempt(session, attempt, f"failed to download result: {exc}")
            kcap.terminal(task_id=attempt.seedance_task_id, status="failed", error_class="download_failed",
                          payload={"stage": "download"}, site="T5")
            return "failed"

        too_short = _result_too_short(attempt.local_clip_path, result_dst)
        attempt.seedance_result_url = url
        attempt.local_result_path = result_dst
        attempt.cost_credits = _parse_cost_credits(data)
        if too_short is not None:
            log.warning(
                "attempt %s: model returned %.2fs for a %.2fs clip (%.1fs "
                "short) — accepting it (same rule as a normal run)",
                attempt.id, too_short.result_sec, too_short.clip_sec,
                too_short.shortfall_sec,
            )
        transition(attempt, RangeAttemptStatus.stitching)
        session.commit()
        # Knowledge capture B4 (AICP-299), after the commit; with preservation on
        # the result is hardlinked before any claim cleanup. Never raises.
        kcap.output(
            task_id=attempt.seedance_task_id, path=result_dst, run_id=run.id, rs_id=rs.id,
            status="short_result" if too_short is not None else "succeeded", payload={"url": url},
            attempt_kind="range_attempt", origin="operator", target_kind="range_attempt",
            target_id=attempt.id, site="B4",
        )

        # ---- Re-cut moved `keep` neighbour(s) from the ORIGINAL source ----
        _assert_still_owner(session, attempt_id, claim_token)
        source = project.source_local_path
        candidate_by_id = {e["id"]: e for e in attempt.candidate_segment_snapshot}
        recut_paths = dict(attempt.keep_recut_paths or {})
        try:
            for sd_id in list(recut_paths.keys()):
                entry = candidate_by_id[sd_id]
                dst = os.path.join(claim_dir, f"keep_{sd_id}.mp4")
                media_mod.cut_clip(
                    source, entry["start_sec"], entry["end_sec"], dst,
                    fast_intermediate=True,
                )
                recut_paths[sd_id] = dst
            attempt.keep_recut_paths = recut_paths
            session.commit()
        except (StaleDataError, _ClaimLost):
            raise
        except Exception as exc:
            _fail_attempt(session, attempt, f"failed to re-cut adjacent keep footage: {exc}")
            return "failed"

        # ---- Candidate stitch (claim_token: keep-cache publish is atomic
        # and ownership-checked — see _resolve_keep_clip) ----
        _assert_still_owner(session, attempt_id, claim_token)
        rs_map = {
            "target_segment_def_id": rs.segment_def_id,
            "by_segment_def_id": {r.segment_def_id: r for r in run.run_segments},
        }
        try:
            _stitch_candidate(
                session, run, project, attempt, rs_map,
                work_dir=claim_dir, claim_token=claim_token,
            )
            session.commit()
        except (StaleDataError, _ClaimLost):
            raise
        except Exception as exc:
            log.exception("poll_range_attempt_once: stitch failed for attempt %s", attempt_id)
            _fail_attempt(session, attempt, f"candidate stitch failed: {_truncate_error(exc)}")
            return "failed"

        # ---- Deliver (only when there IS a candidate final to deliver) ----
        # A run that already has a Drive file is delivered IN PLACE inside
        # _promote, under the Run-row lock (AICP-349) — never a fresh upload
        # here, which would move the Creative Link. Only a run with no Drive
        # file yet gets a new upload, as before.
        if attempt.final_local_path:
            _assert_still_owner(session, attempt_id, claim_token)
            transition(attempt, RangeAttemptStatus.delivering)
            session.commit()
            folder_id = run.gdrive_folder_id or None
            from .config import settings as _settings
            folder_id = folder_id or _settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id and not run.result_gdrive_file_id:
                if gdrive is None:
                    gdrive = GDriveClient()
                try:
                    _assert_still_owner(session, attempt_id, claim_token)
                    result = _deliver_with_retry(gdrive, attempt.final_local_path, folder_id, run.id)
                    attempt.gdrive_file_id = result.get("id")
                    session.commit()
                except (StaleDataError, _ClaimLost):
                    raise
                except Exception as exc:
                    log.exception("poll_range_attempt_once: delivery failed for attempt %s", attempt_id)
                    _fail_attempt(session, attempt, f"candidate delivery failed: {_truncate_error(exc)}")
                    return "failed"

        # ---- Promote (ownership re-checked inside _promote itself) ----
        try:
            _promote(session, run, rs, attempt, claim_token=claim_token,
                     claim_dir=claim_dir, gdrive=gdrive)
        except (StaleDataError, _ClaimLost):
            raise
        except _PromoteUncommitted as exc:
            log.error("poll_range_attempt_once: attempt %s: %s", attempt_id, exc)
            _requeue_uncommitted_promotion(session, attempt_id, claim_token, str(exc))
            return "pending"
        except Exception as exc:
            log.exception("poll_range_attempt_once: promotion failed for attempt %s", attempt_id)
            # _promote mutates `run`/`rs` in-place BEFORE its own commit, so a
            # failure partway through (a state-machine violation, a
            # concurrent-attempt guard, or anything else) leaves those
            # mutations pending on THIS session. Roll back before failing the
            # attempt, or _fail_attempt's commit would flush the very Run/
            # RunSegment changes a "failed" attempt must never apply —
            # exactly the non-destructive-promotion invariant this whole
            # module exists to guarantee.
            session.rollback()
            attempt = session.get(RunSegmentRangeAttempt, attempt_id)
            if attempt is not None and attempt.status == RangeAttemptStatus.done:
                # Not a real failure: _promote's OWN "already promoted"
                # RuntimeError (run.segment_snapshot no longer matches this
                # attempt's base) fires when ANOTHER session's _promote already
                # succeeded — genuinely concurrent processing of the SAME
                # attempt that both reached this final gate without either
                # commit racing the other's version (version_id only protects
                # the earlier, per-field commits above; this is the one gate
                # optimistic versioning cannot see, because the two `_promote`
                # calls are never writing the SAME row version — the run row
                # is what actually serializes them, via the FOR UPDATE lock
                # inside `_promote`). The fresh re-read already shows the
                # correct, successful outcome — report it instead of stamping
                # a misleading error_message onto a row that actually
                # succeeded (noted by the follow-up Cursor review gate as a
                # residual cosmetic gap, not a correctness bug).
                log.info(
                    "poll_range_attempt_once: attempt %s was already promoted "
                    "by another session by the time this one reached _promote "
                    "— reporting its real outcome instead of failing it",
                    attempt_id,
                )
                return "done"
            _fail_attempt(session, attempt, _truncate_error(exc))
            return "failed"
        if published is not None and attempt.final_local_path:
            # AICP-299 §4: this session's own promotion is committed (_promote).
            published.append(PublishedFinal(
                run_id=run.id, project_id=run.project_id, path=attempt.final_local_path, kind="range",
                publication_id=f"range_attempt:{attempt.id}"))
        return "done"
    except (StaleDataError, _ClaimLost):
        # This claim was superseded (its lease expired and a takeover
        # committed, OR an explicit _assert_still_owner re-check caught it)
        # partway through the finalize phase. Discard only THIS claim's own
        # working directory — never a sibling claim's, and never the
        # run/attempt DB state, which the outer poll_range_attempt_once's
        # own except handler reports on from a fresh read.
        _cleanup_claim(claim_dir)
        raise
