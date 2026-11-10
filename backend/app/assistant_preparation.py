"""One explicit operator preparation: repeat final picture, preserve all AAC.

Readable originals remain immutable and cannot execute. Native work is staged
outside SQL locks; one deterministic child and immutable request receipts are
finalized under conversation -> Run locks. No model, provider or queue call.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

from sqlalchemy import select, text

from . import assistant_media as media
from .assistant_access import AssistantAccessDenied, canonical_uuid
from .assistant_models import (AssistantArtifact, AssistantMessage, AssistantSelection,
    AssistantVersion, AssistantVersionArtifact)
from .assistant_store import (ArtifactInput, AssistantConflict, AssistantStoreError,
    _lock, _snapshot, _reference, _selection, _advance_selection,
    append_message, get_version_artifact, verify_artifact)

KIND = "audio_tail_preparation_v1"
MAX_BYTES = 256 * 1024 * 1024


class PreparationCapabilityError(AssistantStoreError):
    pass


def needs_preparation(artifact):
    return isinstance(artifact.metadata_json, dict) and artifact.metadata_json.get("needs_editing_preparation") is True


def require_editable_artifact(session, artifact):
    if artifact is None:
        raise AssistantAccessDenied("assistant access denied")
    owner_full = get_version_artifact(session, artifact.owner_version_id)
    if needs_preparation(artifact) or needs_preparation(owner_full):
        raise PreparationCapabilityError("Prepare an editable copy before editing or publishing this version")


def safe_preparation(artifact):
    if not needs_preparation(artifact):
        return None
    data = artifact.metadata_json.get("editing_preparation")
    if not isinstance(data, dict) or data.get("kind") not in ("audio_tail", "audio_tail_grid_gap"):
        raise PreparationCapabilityError("This version cannot be prepared")
    keys = ("kind", "added_frames", "added_duration_sec", "target_duration_sec", "audio_copied")
    if data["kind"] == "audio_tail_grid_gap":
        keys += ("gap_fill_frames", "tail_hold_frames", "gap_start_sec", "source_declared_duration_sec",
            "source_final_pts_sec", "output_frame_count")
    return {key: data[key] for key in keys}


def prepared_child(session, source_version_id):
    return session.scalar(select(AssistantVersion).where(AssistantVersion.base_version_id == source_version_id,
        AssistantVersion.idempotency_key == "audio-tail:" + source_version_id, AssistantVersion.status == "succeeded"))


def preparation_receipts(session, conversation_id):
    rows = session.scalars(select(AssistantMessage).where(AssistantMessage.conversation_id == conversation_id,
        AssistantMessage.role == "system", AssistantMessage.idempotency_key.like("prepare-request:%"))
        .order_by(AssistantMessage.sequence.desc()).limit(100)).all()
    return [{"idempotency_key": row.context_snapshot["request"]["idempotency_key"],
        "source_version_id": row.context_snapshot["request"]["source_version_id"],
        "version_id": row.context_snapshot["version_id"], "created_at": row.created_at.isoformat()}
        for row in rows if row.context_snapshot.get("kind") == KIND]


def _audio_proof(path, budget):
    raw = media._run([media.FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_streams", "-show_packets",
        "-show_data_hash", "sha256", "-of", "json", str(path)], budget, inputs=(path,))
    try:
        data = json.loads(raw)
        stream, = data["streams"]
        clock = Fraction(stream["time_base"])
        fields = ("codec_name", "profile", "codec_tag_string", "sample_fmt", "sample_rate", "channels",
            "channel_layout", "extradata_hash", "duration_ts", "start_pts")
        description = {key: stream.get(key) for key in fields}
        packets = tuple((str(Fraction(packet["pts"]) * clock), str(Fraction(packet["dts"]) * clock),
            str(Fraction(packet["duration"]) * clock), packet["size"], packet["data_hash"], packet.get("side_data_list", []))
            for packet in data["packets"])
        if not packets or not stream.get("extradata_hash"):
            raise ValueError("missing AAC packet proof")
        return description, packets
    except (KeyError, ValueError, TypeError) as error:
        raise media.AssistantMediaError("AAC packet identity could not be established") from error



def _video_packets(path, budget, *, ignore_editlist=False):
    options = ["-ignore_editlist", "1"] if ignore_editlist else []
    data = json.loads(media._run([media.FFPROBE, "-v", "error", *options, "-select_streams", "v:0",
        "-show_streams", "-show_packets", "-show_data_hash", "sha256", "-of", "json", str(path)],
        budget, inputs=(path,)))
    stream, = data["streams"]
    return stream, data["packets"]


def _decoded_picture(path, budget, *, ignore_editlist=False):
    try:
        options = ["-ignore_editlist", "1"] if ignore_editlist else []
        data = json.loads(media._run([media.FFPROBE, "-v", "error", *options, "-select_streams", "v:0", "-show_streams",
            "-show_frames", "-show_entries", "frame=best_effort_timestamp,pkt_duration,duration:stream=time_base",
            "-of", "json", str(path)], budget, inputs=(path,)))
        stream, = data["streams"]
        clock = Fraction(stream["time_base"])
        return [(Fraction(frame["best_effort_timestamp"])*clock,
            Fraction(frame.get("duration", frame.get("pkt_duration")))*clock) for frame in data["frames"]]
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        raise media.AssistantMediaError("Decoded picture timing proof failed") from error


def _recover_video_clock(path, timing, fps, budget):
    """Prove video-only edit-list bypass recovers exactly the same packets.

    AAC continues through a separate ordinary demux input. No arbitrary MOV
    repair: every packet payload/duration and both clocks must differ only by
    one constant offset, and every source picture must actually decode.
    """
    try:
        normal, before = _video_packets(path, budget)
        ignored, after = _video_packets(path, budget, ignore_editlist=True)
        clock = Fraction(normal["time_base"])
        fields = ("codec_name", "profile", "pix_fmt", "width", "height", "sample_aspect_ratio", "extradata_hash", "time_base")
        if (any(normal.get(field) != ignored.get(field) for field in fields) or not normal.get("extradata_hash")
                or len(before) != timing["source_video_frames"] or len(after) != len(before)):
            raise ValueError("video packet identity changed")
        shift = Fraction(after[0]["pts"]) - Fraction(before[0]["pts"])
        if shift.denominator != 1:
            raise ValueError("video shift is not integral in its native clock")
        if not 0 <= shift*clock <= 1:
            raise ValueError("video recovery offset exceeds the one-second preparation bound")
        for first, second in zip(before, after):
            if (not first.get("data_hash") or first["data_hash"] != second.get("data_hash")
                    or first["size"] != second["size"] or first["duration"] != second["duration"]
                    or Fraction(second["pts"])-Fraction(first["pts"]) != shift
                    or Fraction(second["dts"])-Fraction(first["dts"]) != shift):
                raise ValueError("video demux did not preserve one uniform clock")
        expected = sorted(Fraction(packet["pts"])*clock for packet in before)
        decoded = _decoded_picture(path, budget, ignore_editlist=True)
        if (len(decoded) != len(before) or sorted(point-shift*clock for point, _ in decoded) != expected
                or any(duration != 1/fps for _, duration in decoded)):
            raise ValueError("video decode did not retain every original picture")
        return int(shift), clock
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        raise media.AssistantMediaError("Late-final-frame recovery proof failed") from error


def prepare_audio_tail(source_path, destination_path, *, timeout_sec=240, cancelled=None):
    if type(timeout_sec) not in (int, float) or not 0 < timeout_sec <= 240:
        raise AssistantStoreError("Preparation deadline must be within (0,240] seconds")
    budget = media._budget(timeout_sec, cancelled)
    source, destination = media._input(source_path), media._destination(destination_path)
    try:
        media._probe(source, budget)
    except media.EditingPreparationRequired as error:
        info, timing = error.info, error.preparation
    else:
        raise PreparationCapabilityError("This version does not require preparation")
    if source.stat().st_size > MAX_BYTES:
        raise PreparationCapabilityError("Preparation supports media up to256MiB")
    before_hash = media._hash(source, budget)
    before_audio = _audio_proof(source, budget)
    # The44.1kHz source clock is retained. No-shortest, no audio filters, no
    # global-t or48kHz movie clock: every original AAC packet must survive.
    if timing["kind"] == "audio_tail_grid_gap":
        fps = Fraction(info.fps)
        shift, clock = _recover_video_clock(source, timing, fps, budget)
        # tpad MUST precede setpts: rebasing first can clear the native frame
        # rate used by tpad, losing wanted EOF holds. Extra private padding
        # guards EOF; video-only trim removes it without touching AAC.
        padding = timing["added_frames"] + math.ceil(abs(shift*clock)*fps) + 2
        filters = (f"tpad=stop_mode=clone:stop={padding},setpts=PTS-({shift}),"
            f"fps=fps={info.fps}:round=down:start_time=0:eof_action=pass,trim=end_frame={timing['output_frame_count']}")
        inputs = ["-copyts", "-ignore_editlist", "1", "-noautorotate", "-i", str(source), "-i", str(source)]
        audio_map = "1:a:0"
    else:
        filters = f"tpad=stop_mode=clone:stop={timing['added_frames']}"
        inputs, audio_map = ["-noautorotate", "-i", str(source)], "0:a:0"
    media._run([media.FFMPEG, "-v", "error", "-nostdin", "-n", "-filter_threads", "1", *inputs,
        "-map", "0:v:0", "-map", audio_map, "-vf", filters,
        *media._video_options(info.fps), "-c:a", "copy", "-map_metadata", "-1", "-movflags", "+faststart", str(destination)],
        budget, inputs=(source,), outputs=(destination,))
    try:
        if destination.stat().st_size > MAX_BYTES:
            raise PreparationCapabilityError("Prepared output exceeds256MiB")
        output = media._probe(destination, budget)
        if (output.width, output.height, output.fps, output.sample_aspect_ratio, output.has_audio) != (
                info.width, info.height, info.fps, info.sample_aspect_ratio, True):
            raise media.AssistantMediaError("Preparation changed media geometry or audio presence")
        expected_frames = timing["source_video_frames"] + timing["added_frames"]
        if abs(output.video_duration_sec * float(Fraction(output.fps)) - expected_frames) > 1e-6:
            raise media.AssistantMediaError("Preparation did not append the exact final frames")
        if timing["kind"] == "audio_tail_grid_gap":
            decoded = _decoded_picture(destination, budget)
            fps = Fraction(info.fps)
            if decoded != [(Fraction(index,1)/fps, 1/fps) for index in range(expected_frames)]:
                raise media.AssistantMediaError("Prepared picture grid did not retain full final-frame duration")
        if _audio_proof(destination, budget) != before_audio:
            raise media.AssistantMediaError("Preparation changed AAC packets, timestamps or priming")
        if media._hash(source, budget) != before_hash:
            raise media.AssistantMediaError("Original changed during preparation")
        digest = media._hash(destination, budget)
        return media.MediaEditResult(KIND, (media.SourceProvenance(str(source), before_hash),), str(destination), digest,
            output, float(Fraction(expected_frames, 1) / Fraction(info.fps)),
            (media.TimelineSpan(0, info.video_duration_sec, 0),), "stream_copy_all_aac",
            "Subtitle provenance remains unknown; picture re-encoded and final frame repeated.")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _request(source_id, epoch, key, actor):
    from .assistant_store import _key, _actor
    if type(epoch) is not int or epoch < 0:
        raise AssistantStoreError("Invalid expected selection epoch")
    return {"source_version_id": canonical_uuid(source_id), "expected_selection_epoch": epoch,
        "idempotency_key": _key(key), "actor": _actor(actor)}


def _source(session, service, run_id, source_id):
    conversation = service.conversation(session, run_id)
    version = session.get(AssistantVersion, source_id, populate_existing=True)
    if version is None or version.conversation_id != conversation.id or version.status != "succeeded":
        raise AssistantAccessDenied("assistant access denied")
    service.catalog(session).require_target(source_id)
    artifact = get_version_artifact(session, source_id)
    owner = session.get(AssistantVersion, artifact.owner_version_id, populate_existing=True)
    if owner is None or owner.conversation_id != conversation.id or owner.status != "succeeded":
        raise AssistantAccessDenied("assistant access denied")
    if not needs_preparation(artifact) or artifact.media_kind != "video":
        raise PreparationCapabilityError("This version does not require preparation")
    return conversation, artifact


def _receipt(session, conversation_id, request, child, key):
    append_message(session, conversation_id, idempotency_key=key, actor=request["actor"], role="system",
        content="Prepared an editable copy; original and current Run result retained.", base_version_id=request["source_version_id"],
        context_snapshot={"kind": KIND, "request": request, "version_id": child.id})


def prepare_editable_version(session, service, run_id, source_version_id, *, expected_selection_epoch,
                             idempotency_key, timeout_sec=240, actor="operator"):
    from .models import Run
    from .assistant_run_result import ensure_no_unresolved_intent
    if type(timeout_sec) not in (int, float) or not 0 < timeout_sec <= 240:
        raise AssistantStoreError("Preparation deadline must be within (0,240] seconds")
    budget = media._budget(timeout_sec, None)
    request = _request(source_version_id, expected_selection_epoch, idempotency_key, actor)
    source_id = request["source_version_id"]
    receipt_key = "prepare-request:" + hashlib.sha256(idempotency_key.encode()).hexdigest()
    conversation, artifact = _source(session, service, run_id, source_id)
    conversation_id = conversation.id
    old = session.scalar(select(AssistantMessage).where(AssistantMessage.conversation_id == conversation_id,
        AssistantMessage.idempotency_key == receipt_key))
    if old and old.context_snapshot.get("request") != request:
        raise AssistantConflict("Preparation key payload changed")
    child = session.get(AssistantVersion, old.context_snapshot["version_id"]) if old else prepared_child(session, source_id)
    if child:
        # Replay/alias finalizers still reauthorize under the same lock order;
        # they never reselect, re-render or bypass unresolved publication.
        if session.get_bind().dialect.name == "postgresql":
            session.execute(text("SET LOCAL lock_timeout = '5s'"))
        _lock(session, conversation_id)
        run = service.require_run(session, run_id)
        session.refresh(run, with_for_update=True)
        ensure_no_unresolved_intent(session, run.id)
        _, artifact = _source(session, service, run_id, source_id)
        verify_artifact(artifact, storage=service.storage, cancelled=budget.check)
        verify_artifact(get_version_artifact(session, child.id), storage=service.storage, cancelled=budget.check)
        _receipt(session, conversation_id, request, child, receipt_key)
        session.commit()
        return child
    selection = session.get(AssistantSelection, conversation_id, populate_existing=True)
    if not selection or (selection.selected_version_id, selection.epoch) != (source_id, expected_selection_epoch):
        raise AssistantConflict("Selection changed before preparation")
    ensure_no_unresolved_intent(session, run_id)
    source_path = verify_artifact(artifact, storage=service.storage, cancelled=budget.check)
    source_identity = (artifact.id, artifact.sha256, artifact.size_bytes)
    try:
        media.probe_trusted_media(source_path, timeout_sec=min(30,budget.deadline-time.monotonic()),cancelled=budget.check)
    except media.EditingPreparationRequired as inspection:
        if inspection.preparation != artifact.metadata_json.get("editing_preparation"):
            raise AssistantConflict("Frozen preparation descriptor differs from verified source")
    else:
        raise PreparationCapabilityError("This version no longer requires preparation")
    if artifact.size_bytes > MAX_BYTES:
        raise PreparationCapabilityError("Preparation supports media up to256MiB")
    session.commit()  # No SQL locks held during bounded native work.
    try:
        with tempfile.TemporaryDirectory(prefix="prepare-editable-", dir=service.storage.root / "staging") as directory:
            path = Path(directory) / "prepared.mp4"
            result = prepare_audio_tail(source_path, path, timeout_sec=max(.001, budget.deadline-time.monotonic()))
            output_pin = service.storage.pin(path, cancelled=budget.check)
            if (output_pin.sha256, output_pin.size_bytes) != (result.output_sha256, path.stat().st_size):
                raise AssistantConflict("Prepared output identity changed")
            if session.get_bind().dialect.name == "postgresql":
                session.execute(text("SET LOCAL lock_timeout = '5s'"))
            _lock(session, conversation_id)
            run = service.require_run(session, run_id)
            session.refresh(run, with_for_update=True)
            ensure_no_unresolved_intent(session, run.id)
            if getattr(run.status, "value", run.status) != "done":
                raise AssistantConflict("Run status changed during preparation")
            _, current_artifact = _source(session, service, run_id, source_id)
            if (current_artifact.id, current_artifact.sha256, current_artifact.size_bytes) != source_identity:
                raise AssistantConflict("Preparation source identity changed")
            verify_artifact(current_artifact, storage=service.storage, cancelled=budget.check)
            existing = prepared_child(session, source_id)
            if existing:
                verify_artifact(get_version_artifact(session, existing.id), storage=service.storage, cancelled=budget.check)
                _receipt(session, conversation_id, request, existing, receipt_key)
                session.commit()
                return existing
            selection = _selection(session, conversation_id)
            if (selection.selected_version_id, selection.epoch) != (source_id, expected_selection_epoch):
                raise AssistantConflict("Selection changed during preparation")
            budget.check()
            info = result.metadata
            if media._hash(path, budget) != result.output_sha256:
                raise AssistantConflict("Prepared output changed before publication")
            conversation = _lock(session, conversation_id)
            source_metadata = current_artifact.metadata_json
            recipe, digest = _snapshot({"kind": KIND, "schema_version": 1, "source_version_id": source_id,
                "source_artifact_id": current_artifact.id, "source_sha256": current_artifact.sha256,
                "source_size_bytes": current_artifact.size_bytes, "preparation": source_metadata["editing_preparation"],
                "audio_policy": "stream_copy_all_aac", "picture_policy": ("reencode_fill_gap_retain_late_final_hold_tail"
                    if source_metadata["editing_preparation"]["kind"] == "audio_tail_grid_gap" else "reencode_repeat_final_frame")})
            child = AssistantVersion(id=str(uuid.uuid4()), conversation_id=conversation_id,
                sequence=conversation.next_version_sequence, parent_version_id=source_id, base_version_id=source_id,
                selection_epoch=selection.epoch, idempotency_key="audio-tail:" + source_id, actor=actor,
                request_sha256=digest, recipe=recipe, status="pending")
            session.add(child)
            session.flush()
            metadata = {"duration_sec": info.video_duration_sec, "fps": info.fps, "has_audio": True,
                "preview_mode": "full_as_preview", "subtitle_status": "unknown", "preparation_source_version_id": source_id}
            output = AssistantArtifact(id=str(uuid.uuid4()), owner_version_id=child.id, storage_key=output_pin.storage_key,
                sha256=output_pin.sha256, size_bytes=output_pin.size_bytes, media_kind="video", metadata_json=metadata)
            session.add(output)
            session.flush()
            _reference(session, child.id, output.id, "full")
            _reference(session, child.id, output.id, "preview")
            _reference(session, child.id, current_artifact.id, "input", 0)
            from .assistant_models import _now
            child.status, child.completed_at = "succeeded", _now()
            conversation.next_version_sequence += 1
            session.flush()
            _advance_selection(session, child)
            _receipt(session, conversation_id, request, child, receipt_key)
            budget.check()
            session.commit()
            return child
    except BaseException:
        session.rollback()
        raise
