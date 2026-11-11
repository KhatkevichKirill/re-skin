"""
pipeline_captions.py — process_caption(caption_id) for the "Add Captions" tool.

Re-transcribes a done Run's *delivered* result video (not the project
source — see docs/burned-captions.md §1.1), renders the spoken lines into an
ASS subtitle file (:func:`app.captions.build_ass`), burns them in with ffmpeg
(:func:`app.media.burn_subtitles`), and delivers the captioned copy to
Google Drive as a new derived creative alongside the existing "Add to Final
Creatives" button, which this feature does not touch.

The video actually transcribed/burned is resolved by :func:`_resolve_source`,
not always ``run.result_local_path`` — a caption job can instead be built
from a tail-edited derived creative (``source_type == "tail_edit"``), so
captions can span the whole re-cut video, new tail included, per
docs/burned-captions.md §3.7 (chaining derived creatives). Either way,
``caption.run_id`` stays the *parent Run* (provenance for the project id and
the output filename), never the video being read.

process_caption(caption_id, *, gdrive=None)
    Transitions: queued -> transcribing -> burning -> delivering -> done
    (or -> failed at any point). transcribing is SKIPPED — queued goes
    straight to burning — when ``lines_source == "manual"`` and ``lines`` is
    non-empty: an operator has supplied or hand-edited the caption text via
    ``PATCH /captions/{id}/lines``, and re-transcribing over that edit would
    silently discard it. A non-queued starting status is treated as an
    orphan resume (a prior worker attempt was interrupted) and reset to
    queued first, mirroring pipeline_tail.process_tail_edit's orphan-resume
    handling.

This module intentionally mirrors pipeline_tail.py's shape (single ffmpeg
pass + Drive delivery, duplicated retry/error-truncation helpers in
miniature) rather than sharing code with it — the two pipelines' entities,
validation and phase counts differ enough (this one has an extra
transcribing phase, a language-resolution rule, and a "wordless creative is
a success" branch that a tail edit has no equivalent of) that a shared
abstraction would be net-negative for two small, easy-to-audit pipelines.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional

from sqlalchemy import or_

from . import caption_placement as caption_placement_mod
from . import captions as captions_mod
from . import localisation as localisation_mod
from . import media as media_mod
from . import paid_attempts
from . import spend as spend_mod
from .config import settings
from .paid_attempts_models import OWNER_LLM_CAPTION_ASR
from .db import get_session
from .gdrive_client import GDriveClient
from .models import Run, RunCaption, RunTailEdit
from .state_machine import CaptionStatus, TailEditStatus, transition
from .storage import caption_dir

log = logging.getLogger(__name__)

# Delivery (Google Drive upload) retry tuning — mirrors pipeline_tail's
# TAIL_EDIT_DELIVER_ATTEMPTS / TAIL_EDIT_DELIVER_BACKOFF_SEC so a transient
# Drive error self-heals the same way a tail edit's delivery does.
CAPTION_DELIVER_ATTEMPTS = int(os.getenv("CAPTION_DELIVER_ATTEMPTS", "3"))
CAPTION_DELIVER_BACKOFF_SEC = float(os.getenv("CAPTION_DELIVER_BACKOFF_SEC", "10"))

# How long a caption job waits for an OLDER caption with the same audio that
# is transcribing right now, instead of paying for the same transcription in
# parallel (an operator captioning a project's runs back to back puts two of
# them on the two workers at once). Past it, the job transcribes itself.
CAPTION_DONOR_WAIT_SEC = float(os.getenv("CAPTION_DONOR_WAIT_SEC", "300"))
CAPTION_DONOR_POLL_SEC = float(os.getenv("CAPTION_DONOR_POLL_SEC", "5"))
CAPTION_UNFINGERPRINTED_WAIT_SEC = float(
    os.getenv("CAPTION_UNFINGERPRINTED_WAIT_SEC", "30")
)

# Cap how much of an exception message is persisted to RunCaption.error_message.
_ERROR_MESSAGE_MAX_LEN = 2000

# RunCaption.lines_source values (docs/burned-captions.md §2.1). "manual"
# means an operator supplied/edited `lines` through the API and the
# transcribe step MUST be skipped; "transcribe" is what this module sets
# after a successful call to localisation.transcribe_video; "reused" means
# the lines were copied from another caption with the same audio fingerprint
# (_find_transcript_donor) and no transcription was made.
LINES_SOURCE_MANUAL = "manual"
LINES_SOURCE_TRANSCRIBE = "transcribe"
LINES_SOURCE_REUSED = "reused"


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


def _find_transcript_donor(
    caption: RunCaption, audio_sha256: str, language: Optional[str], session
) -> Optional[RunCaption]:
    """Return another caption whose source had the same decoded audio, or None.

    Every face_swap run of one project under ``audio_mode="original"``
    delivers a bit-identical soundtrack, so its transcript is the same one;
    copying it saves a Gemini call per run and keeps the captions identical
    across the project's creatives instead of jittering by a few hundred ms.

    Only rows with real lines qualify (an empty transcript is not reused — a
    wordless result costs little to confirm and could be a model miss). An
    operator's edit (``manual``) wins over a transcription so a correction
    made on one creative reaches the next ones; otherwise the newest row.
    When *language* is known it must match: captions render per language.
    """
    rows = (
        session.query(RunCaption)
        .filter(
            RunCaption.audio_sha256 == audio_sha256,
            RunCaption.id != caption.id,
            RunCaption.lines_source.isnot(None),
        )
        # Re-read rows already in the identity map: the caller polls while
        # another worker writes them.
        .execution_options(populate_existing=True)
        .all()
    )
    candidates = [
        r for r in rows
        if r.lines and (language is None or r.language == language)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (r.lines_source == LINES_SOURCE_MANUAL, r.updated_at),
        reverse=True,
    )
    return candidates[0]


def _older_sibling_transcribing(
    caption: RunCaption, audio_sha256: str, session
) -> Optional[str]:
    """What an OLDER caption mid-transcription means for this job's audio.

    ``"same"`` when one has this fingerprint; ``"unknown"`` when one has no
    fingerprint yet — it may be the same audio, still between its
    ``transcribing`` commit and its fingerprint commit (a second or two of
    probe + decode); ``None`` when neither. Only older ones, so two jobs that
    start together never wait on each other: the newer waits, the older
    transcribes.
    """
    rows = (
        session.query(RunCaption.audio_sha256)
        .filter(
            RunCaption.id != caption.id,
            RunCaption.status == CaptionStatus.transcribing,
            RunCaption.created_at < caption.created_at,
            or_(
                RunCaption.audio_sha256 == audio_sha256,
                RunCaption.audio_sha256.is_(None),
            ),
        )
        .all()
    )
    if any(fp == audio_sha256 for (fp,) in rows):
        return "same"
    return "unknown" if rows else None


def _await_transcript_donor(
    caption: RunCaption, audio_sha256: str, language: Optional[str], session
) -> Optional[RunCaption]:
    """_find_transcript_donor, waiting up to CAPTION_DONOR_WAIT_SEC for an
    older same-audio caption that is transcribing right now to finish.

    An older caption that has not fingerprinted yet is waited on for at most
    CAPTION_UNFINGERPRINTED_WAIT_SEC — long enough for its fingerprint to
    land, short enough that an audio-less caption (it never gets one) does
    not hold this job for its whole transcription.
    """
    start = time.monotonic()
    waited = False
    while True:
        donor = _find_transcript_donor(caption, audio_sha256, language, session)
        if donor is not None:
            return donor
        sibling = _older_sibling_transcribing(caption, audio_sha256, session)
        elapsed = time.monotonic() - start
        if sibling is None:
            break
        if sibling == "unknown" and elapsed >= CAPTION_UNFINGERPRINTED_WAIT_SEC:
            break
        if elapsed >= CAPTION_DONOR_WAIT_SEC:
            log.info(
                "caption_id=%s: gave up waiting %.0fs for a same-audio caption "
                "to finish transcribing", caption.id, CAPTION_DONOR_WAIT_SEC,
            )
            break
        if not waited:
            log.info(
                "caption_id=%s: an older caption (%s audio) is transcribing — "
                "waiting for its lines", caption.id, sibling,
            )
            waited = True
        # End the read transaction before sleeping so no snapshot or
        # "idle in transaction" connection is held across the wait.
        session.commit()
        time.sleep(CAPTION_DONOR_POLL_SEC)
    return None


def _truncate_error(exc: Exception) -> str:
    """Render *exc* as a bounded-length string safe to store in error_message."""
    msg = str(exc)
    if len(msg) > _ERROR_MESSAGE_MAX_LEN:
        msg = msg[:_ERROR_MESSAGE_MAX_LEN] + "... (truncated)"
    return msg


def _resolve_source_tail_edit(caption: RunCaption, session) -> Optional[RunTailEdit]:
    """Fetch + validate the RunTailEdit a chained caption job reads from.

    Returns None when ``caption.source_type != "tail_edit"`` (the unchained,
    default case). Raises ValueError — the same shape as the existing "run
    has no result video on disk" failure — when the referenced tail edit is
    missing, is not ``done``, or (checked by the caller) has no file on
    disk. Split out from :func:`_resolve_source` so the output-filename step
    can re-fetch the SAME already-validated tail edit (a cheap identity-map
    hit on the session already used to fetch it) to read its own clip-name
    segment, without duplicating this validation.
    """
    if caption.source_type != "tail_edit":
        return None
    if not caption.source_tail_edit_id:
        raise ValueError(
            "caption source_type='tail_edit' but source_tail_edit_id is unset"
        )
    edit = session.get(RunTailEdit, caption.source_tail_edit_id)
    if edit is None:
        raise ValueError(f"source tail edit {caption.source_tail_edit_id!r} not found")
    status_val = (
        edit.status.value if hasattr(edit.status, "value") else str(edit.status)
    )
    if status_val != TailEditStatus.done.value:
        raise ValueError(
            f"source tail edit {caption.source_tail_edit_id!r} is not done "
            f"(status={status_val})"
        )
    return edit


def _resolve_source(caption: RunCaption, run: Run, session) -> str:
    """Return the absolute path to the video this caption job transcribes
    and burns onto.

    ``caption.run_id`` (== ``run.id``) is always the parent Run — the
    provenance behind the output filename, not necessarily the video being
    read. ``caption.source_type`` says which file that is:
      "run"       (default) -> ``run.result_local_path``, byte-for-byte the
                  v1 behaviour.
      "tail_edit" -> the referenced RunTailEdit's ``output_path`` (captions
                  span the whole re-cut video, new tail included — see
                  docs/burned-captions.md §3.7).
    Mirrors overlay_pipeline._resolve_source / pipeline_tail._resolve_source.
    Raises ValueError when the referenced tail edit has no file on disk
    (missing/not-done is already raised by _resolve_source_tail_edit above).
    """
    edit = _resolve_source_tail_edit(caption, session)
    if edit is not None:
        path = edit.output_path
        if not path or not os.path.exists(path):
            raise ValueError("source tail edit has no output video on disk")
        return path

    # source_type == "run" (default) — byte-for-byte the v1 behaviour.
    path = run.result_local_path
    if not path or not os.path.exists(path):
        raise ValueError(f"run {caption.run_id!r} has no result video on disk")
    return path


def _tail_edit_clip_name(edit: RunTailEdit) -> str:
    """Recover the ``{clip_name}`` segment from a tail edit's own output
    filename.

    A tail edit's filename is always ``{run_id}_{clip_name}.mp4`` (or the
    chained ``{run_id}_captioned_{clip_name}.mp4`` — see
    pipeline_tail._output_filename), so stripping the leading ``{run_id}_``
    and the ``.mp4`` suffix recovers exactly what pipeline_tail put there,
    without re-deriving it from the LibraryClip lookup here.
    """
    base = os.path.basename(edit.output_path or "")
    stem = base[:-4] if base.lower().endswith(".mp4") else base
    prefix = f"{edit.run_id}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem or "clip"


def _output_filename(run_id: str, source_tail_edit: Optional[RunTailEdit]) -> str:
    """Output/delivery filename for a caption job: ``{run_id}_captioned.mp4``.

    When this caption's source is a tail edit (chained,
    ``source_tail_edit`` not None), the tail edit's own clip-name segment is
    folded in so a reader can tell which re-cut got captioned, and so the
    filename cannot collide with an unchained caption's:
    ``{run_id}_{clip_name}_captioned.mp4``.
    """
    if source_tail_edit is not None:
        return f"{run_id}_{_tail_edit_clip_name(source_tail_edit)}_captioned.mp4"
    return f"{run_id}_captioned.mp4"


def _deliver_with_retry(
    gdrive: GDriveClient, path: str, folder_id: str, caption_id: str, filename: str
) -> dict:
    """Upload the captioned video to Drive, retrying the whole upload on failure.

    Raises the last exception if every attempt fails (the caller then marks
    the caption failed — but the output file stays on disk, so a manual
    retry re-delivers without re-transcribing or re-burning).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, CAPTION_DELIVER_ATTEMPTS + 1):
        try:
            return gdrive.upload_file(path, folder_id, filename)
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Drive upload attempt %d/%d failed for caption %s: %s",
                attempt, CAPTION_DELIVER_ATTEMPTS, caption_id, exc,
            )
            if attempt < CAPTION_DELIVER_ATTEMPTS:
                time.sleep(CAPTION_DELIVER_BACKOFF_SEC * attempt)
    raise last_exc  # type: ignore[misc]


class CaptionFenced(RuntimeError):
    """This job no longer owns the caption's ASR (AICP-351): write nothing."""


def caption_asr_job_timeout_sec() -> int:
    from .tasks import CAPTION_JOB_TIMEOUT

    return int(CAPTION_JOB_TIMEOUT)


def _caption_fence(caption_id: str, token: Optional[str]):
    """True while this job owns the caption's ASR (locks the caption row: lock
    order caption -> attempt)."""

    def fence(db) -> bool:
        cap = db.query(RunCaption).filter(RunCaption.id == caption_id).with_for_update() \
            .populate_existing().one_or_none()
        return (cap is not None and token is not None
                and cap.status == CaptionStatus.transcribing and cap.asr_claim == token)

    return fence


def _transcribe_with_attempts(caption_id: str, token: Optional[str], epoch: int,
                              final_path: str, *, run_id: Optional[str]):
    """The caption's ASR as §5.2 attempts (at most 3 per epoch, only an unknown
    outcome retried). Returns ``(transcript, attempt_id)`` with the attempt
    still ``started`` (accepted by the fenced transcript write); raises
    :class:`CaptionFenced` or ``LocalisationError``."""
    from .localisation_jobs import _llm_attempts

    owner = paid_attempts.caption_owner_id(caption_id, epoch)
    result, aid, err, fenced = _llm_attempts(
        owner_kind=OWNER_LLM_CAPTION_ASR, job_id=owner, token=token,
        sessions=get_session, fence=_caption_fence(caption_id, token),
        max_calls=paid_attempts.TRANSCRIBE_CALLS_PER_ATTEMPT * localisation_mod.transcribe_chain_len(),
        lease_sec=paid_attempts.llm_lease_sec(caption_asr_job_timeout_sec()),
        scope=spend_mod.caption_scope(caption_id), kind="transcribe",
        call=lambda ledger: localisation_mod.transcribe_video(final_path, ledger=ledger),
        run_id=run_id,
    )
    if fenced:
        raise CaptionFenced(f"caption {caption_id}: ASR claim superseded before the transcript")
    if err is not None:
        if err.startswith("interrupted:"):
            err = (err + " — or replace the ASR with captions from the script "
                   "(PATCH the caption's lines).")
        raise localisation_mod.LocalisationError(err)
    return result, aid


def _release_caption_scope(session, caption) -> None:
    """At a caption's terminal write: release its unused reservation once no
    ASR attempt of it can still send (AICP-351 §5.2). Never raises."""
    try:
        with session.begin_nested():
            paid_attempts.release_llm_scope_if_idle(
                session, OWNER_LLM_CAPTION_ASR,
                paid_attempts.caption_owner_id(caption.id, caption.asr_epoch),
                spend_mod.caption_scope(caption.id), kind="captions")
    except Exception:
        log.exception("caption %s: reservation release failed", caption.id)


def asr_reap_decision(session, caption, *, evidence) -> str:
    """The derived reaper's decision on an orphaned caption in ``transcribing``
    (caller's transaction; AICP-351 §5.2): ``requeue`` (the ASR claim replaced
    by a re-dispatch nonce, ``caption.asr_claim = "dispatch:<nonce>"``; the
    caller enqueues with that nonce after its commit), ``wait`` (its attempt's sender
    may still run and its lease has not run out) or ``fail`` (the bound is
    spent: reaped to failed as before)."""
    from datetime import datetime

    session.refresh(caption, with_for_update=True)
    owner = paid_attempts.caption_owner_id(caption.id, caption.asr_epoch)
    now = datetime.utcnow()
    att = paid_attempts.open_attempt(session, OWNER_LLM_CAPTION_ASR, owner, for_update=True)
    if att is not None:
        if paid_attempts.reap_llm_attempt(session, att, now=now, evidence=evidence) is None:
            return "wait"
        session.flush()  # the latest-attempt read below must see this decision
    # Only an UNKNOWN outcome is retried (§5.2): the epoch's latest attempt
    # must have ended unknown / superseded. A worker that died before its
    # first call, or after a definite failure, is reaped to failed as before.
    from .paid_attempts_models import STOPPED, SUPERSEDED, UNKNOWN, PaidSubmitAttempt

    last = session.query(PaidSubmitAttempt).filter(
        PaidSubmitAttempt.owner_kind == OWNER_LLM_CAPTION_ASR,
        PaidSubmitAttempt.owner_id == owner,
    ).order_by(PaidSubmitAttempt.attempt_no.desc()).first()
    # STOPPED = superseded then acknowledged: its outcome is still unknown.
    if (last is None or last.status not in (UNKNOWN, SUPERSEDED, STOPPED)
            or paid_attempts.attempt_count(session, OWNER_LLM_CAPTION_ASR, owner)
            >= paid_attempts.LLM_MAX_ATTEMPTS):
        caption.asr_claim = None
        return "fail"
    caption.asr_claim = DISPATCH_PREFIX + uuid.uuid4().hex
    caption.error_message = "worker gone mid-transcription — retried automatically (AICP-351)"
    return "requeue"


DISPATCH_PREFIX = "dispatch:"


def process_caption(caption_id: str, *, gdrive: Optional[GDriveClient] = None,
                    dispatch: Optional[str] = None) -> None:
    """
    Re-transcribe a done Run's delivered result, burn the lines in, deliver.

    Transitions
    -----------
    queued -> transcribing -> burning -> delivering -> done
    (queued -> burning directly when lines_source == "manual" and lines is
    non-empty; on error: -> failed, error_message set, exception re-raised)

    Validates before any encoding (fails the caption, does not raise
    unhandled):
    - the Run exists;
    - the resolved source (the Run's result, or — when source_type ==
      "tail_edit" — the referenced RunTailEdit's output, per
      _resolve_source) has a video on disk, and (when chained) that the
      referenced tail edit exists and is ``done``.

    A transcript with no lines (a wordless creative) is a SUCCESS, not a
    failure — it produces a caption-free re-encode. A
    localisation.LocalisationError propagates to the outer except block like
    any other failure, and fails the caption with its message.
    """
    t_start = time.monotonic()
    log.info("process_caption start: caption_id=%s", caption_id)

    with get_session() as session:
        caption: RunCaption = session.get(RunCaption, caption_id)
        if caption is None:
            raise ValueError(f"RunCaption not found: {caption_id}")

        asr_token = None
        if dispatch is not None:
            # AICP-351: a reaper re-dispatch runs only while the caption still
            # carries ITS nonce (taken under the row lock); a stale duplicate
            # delivery — or one overtaken by an operator Retry — exits untouched.
            session.refresh(caption, with_for_update=True)
            if caption.asr_claim != DISPATCH_PREFIX + dispatch:
                session.rollback()
                log.warning("process_caption: caption %s re-dispatch %s is stale (claim %r) — "
                            "not run", caption_id, dispatch, caption.asr_claim)
                return
            # Consume the nonce in the SAME transaction as the check: a duplicate
            # delivery of this dispatch then finds the claim changed and exits.
            asr_token = uuid.uuid4().hex
            caption.asr_claim = asr_token
            session.commit()
        try:
            # Orphan-resume: any non-queued starting status means a prior
            # attempt was interrupted mid-flight — reset to queued before
            # advancing (mirrors pipeline_tail.process_tail_edit).
            if caption.status != CaptionStatus.queued:
                log.info(
                    "process_caption: caption %s is in status=%r (not queued) "
                    "— resetting to queued for orphan resume",
                    caption_id, caption.status,
                )
                try:
                    transition(caption, CaptionStatus.queued)
                except Exception:
                    # transcribing/burning/delivering lack a direct ->queued
                    # edge; go via failed.
                    transition(caption, CaptionStatus.failed)
                    transition(caption, CaptionStatus.queued)
                session.commit()

            # An operator-supplied/edited transcript (PATCH /captions/{id}
            # /lines) skips transcription entirely — re-transcribing would
            # silently throw the edit away. This is decided from the row
            # alone, before the Run is even loaded, because it decides which
            # edge out of `queued` this run of the job takes.
            skip_transcribe = (
                caption.lines_source == LINES_SOURCE_MANUAL and bool(caption.lines)
            )
            transition(
                caption,
                CaptionStatus.burning if skip_transcribe else CaptionStatus.transcribing,
            )
            # AICP-351 §5.2: this job's ASR dispatch fence (a fresh token per
            # entry into `transcribing`; the reaper clears it to re-dispatch).
            if not skip_transcribe:
                asr_token = asr_token or uuid.uuid4().hex
                caption.asr_claim = asr_token
            session.commit()

            run: Run = session.get(Run, caption.run_id)
            if run is None:
                raise ValueError(f"Run not found: {caption.run_id}")

            final_path = _resolve_source(caption, run, session)

            # Probed ONCE, reused for build_ass (width/height/duration) and
            # burn_subtitles (duration/has_audio) — burn_subtitles accepts
            # both precisely so it can skip its own probe of the same file.
            # This probes the RESOLVED source, never the Run's result — a
            # chained tail-edit source can be a different duration, and
            # build_ass sizing/clamping the ASS against the wrong length
            # would misplace or drop captions.
            info = media_mod.probe(final_path)

            # ---------------------------------------------------------------
            # Transcribe (skipped for a manual transcript)
            # ---------------------------------------------------------------
            if skip_transcribe:
                caption.language = caption.language or run.language
                # Fingerprinted so an operator's edit can be copied onto the
                # project's next caption jobs (_find_transcript_donor).
                caption.audio_sha256 = media_mod.audio_fingerprint(final_path)
                session.commit()
            else:
                t_transcribe_start = time.monotonic()
                # Fingerprint the RESOLVED source (a chained tail edit has a
                # different tail, hence different audio) and stored even when
                # nothing is reused, so this row can be the next job's donor.
                caption.audio_sha256 = media_mod.audio_fingerprint(final_path)
                caption.lines_reused_from = None
                # Committed BEFORE any wait or Gemini call: a newer same-audio
                # job must see this one is transcribing (and wait for it), and
                # no row lock may be held across a minutes-long remote call.
                session.commit()
                donor = None
                if caption.audio_sha256:
                    donor = _await_transcript_donor(
                        caption,
                        caption.audio_sha256,
                        caption.language or run.language,
                        session,
                    )
                # Close the read transaction the lookup opened: nothing may
                # sit "idle in transaction" through a minutes-long Gemini call.
                session.commit()

                if donor is not None:
                    lines = [dict(line) for line in donor.lines]
                    caption.lines = lines
                    caption.lines_source = LINES_SOURCE_REUSED
                    caption.lines_reused_from = donor.id
                    caption.language = (
                        caption.language or run.language or donor.language
                    )
                    log.info(
                        "caption_id=%s: reusing %d line(s) from caption %s "
                        "(%s, same audio %s) — transcription skipped",
                        caption_id, len(lines), donor.id, donor.lines_source,
                        caption.audio_sha256[:12],
                    )
                else:
                    # AICP-351 §5.2: bounded, claimed, single-send attempts;
                    # the transcript write below is fenced by the ASR claim
                    # AND the attempt (a superseded worker writes nothing).
                    transcript, asr_attempt = _transcribe_with_attempts(
                        caption_id, asr_token, int(caption.asr_epoch or 0), final_path,
                        run_id=run.id,
                    )
                    lines = transcript.get("lines") or []
                    session.refresh(caption, with_for_update=True)
                    if (caption.status != CaptionStatus.transcribing
                            or caption.asr_claim != asr_token
                            or not paid_attempts.finish_llm(session, asr_attempt, "accepted")):
                        session.rollback()
                        # Acknowledge the attempt anyway (its last I/O has
                        # returned): a superseded sender must not stay counted
                        # active forever (design §9 item 2).
                        paid_attempts.finish_llm_own_session(asr_attempt, "accepted",
                                                             sessions=get_session)
                        raise CaptionFenced(
                            f"caption {caption_id}: ASR claim superseded — transcript discarded")
                    caption.lines = lines
                    caption.lines_source = LINES_SOURCE_TRANSCRIBE
                    caption.language = (
                        caption.language or run.language or transcript.get("source_language")
                    )
                caption.asr_claim = None
                session.commit()

                if not lines:
                    # A wordless creative is a SUCCESS, not a failure — it
                    # produces a caption-free re-encode below, same as if an
                    # operator had submitted an empty manual transcript.
                    log.info(
                        "caption_id=%s: transcript has no lines (wordless "
                        "creative) — continuing to a caption-free re-encode",
                        caption_id,
                    )
                log.info(
                    "caption_id=%s transcribe_phase=%.1fs lines=%d",
                    caption_id, time.monotonic() - t_transcribe_start, len(lines),
                )

                transition(caption, CaptionStatus.burning)
                session.commit()

            # ---------------------------------------------------------------
            # Build the ASS file + burn
            # ---------------------------------------------------------------
            t_encode_start = time.monotonic()
            out_dir = caption_dir(caption.id, run.project_id)

            ass_path = os.path.join(out_dir, "captions.ass")
            ass_content = captions_mod.build_ass(
                caption.lines or [],
                width=info.width,
                height=info.height,
                language=caption.language,
                duration_sec=info.duration_sec,
                # AICP-368: the placement frozen on the job at creation, checked
                # against the RESOLVED source's frame (a different aspect falls
                # back to the house position). None = house style.
                placement=caption_placement_mod.burn_placement(
                    caption.placement, width=info.width, height=info.height
                ),
            )
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

            # Re-fetch the source tail edit (if any) purely for its filename
            # segment — _resolve_source already validated it via the same
            # session, so this is a free identity-map hit, not a new query.
            source_tail_edit = _resolve_source_tail_edit(caption, session)
            out_filename = _output_filename(run.id, source_tail_edit)
            output_path = os.path.join(out_dir, out_filename)
            media_mod.burn_subtitles(
                final_path,
                ass_path,
                output_path,
                duration_sec=info.duration_sec,
                has_audio=info.has_audio,
            )
            # Normally a no-op (burn_subtitles already caps the output via
            # delivery_video_opts); this catches an edge case the cap could
            # not reach. Never fail the caption over it — an oversized
            # deliverable still beats no deliverable.
            try:
                media_mod.enforce_size_limit(output_path)
            except Exception as exc:
                log.warning(
                    "caption_id=%s: could not shrink %s to the delivery size "
                    "cap: %s",
                    caption_id, output_path, exc,
                )
            caption.style_version = captions_mod.CAPTION_STYLE_VERSION
            caption.output_path = output_path
            session.commit()
            log.info(
                "caption_id=%s encode_phase=%.1fs -> %s",
                caption_id, time.monotonic() - t_encode_start, output_path,
            )

            # ---------------------------------------------------------------
            # Deliver
            # ---------------------------------------------------------------
            t_deliver_start = time.monotonic()
            transition(caption, CaptionStatus.delivering)
            session.commit()
            folder_id = run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id:
                if gdrive is None:
                    gdrive = _default_gdrive()
                result = _deliver_with_retry(
                    gdrive, output_path, folder_id, caption_id, out_filename
                )
                caption.gdrive_file_id = result.get("id")
                log.info(
                    "caption_id=%s deliver_phase=%.1fs gdrive_file=%s",
                    caption_id, time.monotonic() - t_deliver_start, caption.gdrive_file_id,
                )
                session.flush()

            # Clear any stale failure message from a prior attempt.
            caption.error_message = None
            transition(caption, CaptionStatus.done)
            session.flush()
            _release_caption_scope(session, caption)
            session.commit()
            log.info(
                "process_caption done: caption_id=%s total=%.1fs",
                caption_id, time.monotonic() - t_start,
            )

        except CaptionFenced as exc:
            # Another dispatch owns this caption's ASR now: write NOTHING (a
            # failed write here would fail the new owner's job).
            session.rollback()
            log.warning("process_caption: %s", exc)
            return
        except Exception as exc:
            log.exception("process_caption failed: caption_id=%s", caption_id)
            session.rollback()
            caption = session.get(RunCaption, caption_id)
            if caption is not None and asr_token is not None and caption.asr_claim not in (
                    None, asr_token):
                log.warning("process_caption: caption %s re-dispatched meanwhile — failure "
                            "not written", caption_id)
                return
            if caption is not None:
                if caption.status not in (CaptionStatus.done, CaptionStatus.failed):
                    try:
                        transition(caption, CaptionStatus.failed)
                    except Exception:
                        caption.status = CaptionStatus.failed
                caption.error_message = _truncate_error(exc)
                if caption.status == CaptionStatus.failed:
                    caption.asr_claim = None
                    session.flush()
                    _release_caption_scope(session, caption)
            try:
                session.commit()
            except Exception:
                pass
            raise
