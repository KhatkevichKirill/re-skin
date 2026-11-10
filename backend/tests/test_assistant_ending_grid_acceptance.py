"""Independent native proof for the exact late-final-frame MOV profile.

Fixtures patch only an owned MP4 composition-time table. Each picture carries
its own binary identifier, so a plausible frame count cannot hide a dropped,
shifted or substituted original picture. No live media/services are used.
"""
from __future__ import annotations

from fractions import Fraction
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import select

from app import assistant_media as media
from app import assistant_preparation as preparation
from app.assistant_access import ProjectAllowlist
from app.assistant_legacy import import_legacy_run
from app.assistant_models import AssistantRunResultPromotion, AssistantVersionArtifact
from app.assistant_store import get_version_artifact
from app.assistant_turns import AssistantTurnService
from app.models import Run
from tests.test_assistant_store import setup, legacy
from tests.test_assistant_preparation_acceptance import (
    command, probe, sha, audio_identity, pcm, assert_audible_cta_tail,
    native_original, prepare, output, run_snapshot, FakeDrive, append_edit,
)

N = 892
F = Fraction(30)
EXPECTED_SOURCE_SLOTS = list(range(N - 1)) + [N]
EXPECTED_OUTPUT_IDS = list(range(N - 1)) + [N - 2, N - 1, N - 1, N - 1]


def boxes(data, first=0, last=None):
    last = len(data) if last is None else last
    while first < last:
        size, kind = struct.unpack_from(">I4s", data, first)
        assert size >= 8 and first + size <= last, "owned fixture has invalid MP4 atoms"
        yield first, size, kind
        first += size
    assert first == last


def child(data, parent, kind):
    first, size, _ = parent
    return next(item for item in boxes(data, first + 8, first + size) if item[2] == kind)


def add_late_final_composition_offset(source, destination):
    """Move only the final presentation by512 ticks; keep decode duration.

    Media precedes moov in this fixture, so extending its tables requires no
    chunk-offset changes. This reproduces the incident that a normal remux can
    repair accidentally and that a PTS+DTS shift does not reproduce.
    """
    data = source.read_bytes()
    top = list(boxes(data))
    moov = next(item for item in top if item[2] == b"moov")
    assert moov[0] > next(item for item in top if item[2] == b"mdat")[0]
    for trak in boxes(data, moov[0] + 8, moov[0] + moov[1]):
        if trak[2] != b"trak":
            continue
        mdia = child(data, trak, b"mdia")
        handler = child(data, mdia, b"hdlr")
        if data[handler[0] + 16:handler[0] + 20] == b"vide":
            break
    else:
        raise AssertionError("owned fixture has no video track")
    minf = child(data, mdia, b"minf")
    stbl = child(data, minf, b"stbl")
    existing = next((item for item in boxes(data, stbl[0] + 8, stbl[0] + stbl[1]) if item[2] == b"ctts"), None)
    offsets = [0] * N
    if existing:
        first, _, _ = existing
        version, entries = struct.unpack_from(">II", data, first + 8)
        assert version == 0
        offsets = []
        for index in range(entries):
            count, offset = struct.unpack_from(">II", data, first + 16 + 8 * index)
            offsets.extend([offset] * count)
        assert len(offsets) == N
    packets = probe(source, "-select_streams", "v:0", "-show_packets")["packets"]
    final_packet = max(range(len(packets)), key=lambda index: packets[index]["pts"])
    offsets[final_packet] += 512
    runs = []
    for offset in offsets:
        if runs and runs[-1][1] == offset:
            runs[-1][0] += 1
        else:
            runs.append([1, offset])
    replacement = struct.pack(">I4sII", 16 + 8 * len(runs), b"ctts", 0, len(runs))
    replacement += b"".join(struct.pack(">II", *entry) for entry in runs)
    first = existing[0] if existing else stbl[0] + stbl[1]
    old_size = existing[1] if existing else 0
    patched = bytearray(data[:first] + replacement + data[first + old_size:])
    for position, size, _ in (moov, trak, mdia, minf, stbl):
        struct.pack_into(">I", patched, position, size + len(replacement) - old_size)
    destination.write_bytes(patched)


def gap_original(data, *, bframes=2):
    video, audio, normal, original = (data / name for name in ("ids.mp4", "sound.m4a", "normal.mp4", "ending-gap.mp4"))
    # Binary identifiers are read from the top20 pixels; body colour proves
    # the distinct final picture survives beyond the declared stream end.
    filters = ["drawbox=c=blue:t=fill:enable='eq(n,891)'", "drawbox=x=0:y=0:w=120:h=20:c=black:t=fill"]
    filters += [f"drawbox=x={12 * bit}:y=0:w=12:h=20:c=white:t=fill:enable='bitand(n,{1 << bit})'" for bit in range(10)]
    command(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=160x120:r=30",
        "-vf", ",".join(filters), "-frames:v", str(N), "-an", "-c:v", "libx264", "-bf", str(bframes),
        "-threads", "1", "-pix_fmt", "yuv420p", "-video_track_timescale", "15360", str(video)])
    command(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "aevalsrc='0.2*sin(2*PI*if(lt(t,29.70),440,1320)*t)':s=44100",
        "-af", "atrim=end_sample=1315106,asetpts=N/SR/TB", "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-movie_timescale", "44100", str(audio)])
    command(["ffmpeg", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
        "-c", "copy", "-movie_timescale", "44100", "-video_track_timescale", "15360", str(normal)])
    add_late_final_composition_offset(normal, original)
    return original


def picture_ids(path, *, ignore_editlist=False):
    options = ["-ignore_editlist", "1"] if ignore_editlist else []
    raw = command(["ffmpeg", "-v", "error", "-copyts", *options, "-i", str(path), "-map", "0:v:0",
        "-vf", "crop=120:20:0:0,format=gray", "-vsync", "0", "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 20, 120)
    bits = np.stack([frames[:, 2:18, 12 * bit + 2:12 * bit + 10].mean(axis=(1, 2)) > 128 for bit in range(10)], axis=1)
    return (bits * (1 << np.arange(10))).sum(axis=1).tolist()


def final_colours(path):
    raw = command(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf",
        "select=gte(n\\,890),crop=160:100:0:20,format=rgb24", "-vsync", "0", "-f", "rawvideo", "-"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 100, 160, 3).mean(axis=(1, 2))


def input_grid(source):
    stream = probe(source, "-select_streams", "v:0", "-show_streams")["streams"][0]
    packets = probe(source, "-select_streams", "v:0", "-show_packets", "-show_data_hash", "sha256")["packets"]
    clock = Fraction(stream["time_base"])
    assert (stream["avg_frame_rate"], stream["r_frame_rate"], clock, int(stream["nb_frames"])) == ("30/1", "30/1", Fraction(1, 15360), N)
    assert Fraction(stream["duration_ts"]) * clock == Fraction(N, 1) / F
    assert sorted(Fraction(packet["pts"]) * clock * F for packet in packets) == EXPECTED_SOURCE_SLOTS
    assert all(Fraction(packet["duration"]) * clock == 1 / F for packet in packets)
    assert any("D" in packet["flags"] for packet in packets if packet["pts"] == N * 512), "fixture must expose the default edit-list discard trap"
    return stream, packets


@pytest.mark.parametrize("bframes", [0, 2])
def test_owned_input_has_exact_gap_and_ignore_recovers_every_identified_frame(tmp_path, bframes):
    source = gap_original(tmp_path, bframes=bframes)
    stream, normal = input_grid(source)
    ignored = probe(source, "-ignore_editlist", "1", "-select_streams", "v:0", "-show_packets", "-show_data_hash", "sha256")["packets"]
    assert len(normal) == len(ignored) == N
    shift = ignored[0]["pts"] - normal[0]["pts"]
    assert shift == (1024 if bframes else 0)
    assert all(after["pts"] - before["pts"] == shift and after["dts"] - before["dts"] == shift
        and after["data_hash"] == before["data_hash"] for before, after in zip(normal, ignored))
    assert picture_ids(source) == list(range(N - 1)), "normal demux should omit the late final picture in this fixture"
    assert picture_ids(source, ignore_editlist=True) == list(range(N))
    with pytest.raises(media.EditingPreparationRequired) as error:
        media.probe_trusted_media(source)
    timing = error.value.preparation
    assert (timing["kind"], timing["gap_fill_frames"], timing["tail_hold_frames"], timing["output_frame_count"]) == ("audio_tail_grid_gap", 1, 2, 895)


@pytest.fixture
def grid_case(setup):
    sessions, storage, data = setup
    source = gap_original(data)
    run_id, project_id, main = legacy(sessions, data, content=source.read_bytes(), duration=1315106 / 44100)
    service = AssistantTurnService(storage=storage, policy=ProjectAllowlist(project_id), enabled=True)
    with sessions.begin() as session:
        run = session.get(Run, run_id)
        run.result_gdrive_file_id = "existing-drive-id"
        original = import_legacy_run(session, run, actor="operator", storage=storage, policy=service.policy)
        original_id, conversation_id = original.id, original.conversation_id
    return SimpleNamespace(sessions=sessions, storage=storage, data=data, source=main, run_id=run_id,
        project_id=project_id, service=service, original_id=original_id, conversation_id=conversation_id)


def test_prepared_output_preserves_all_original_slots_audio_and_main(grid_case):
    c = grid_case
    input_grid(c.source)
    original_hash, original_main = sha(c.source), run_snapshot(c)
    sound, decoded_sound = audio_identity(c.source), pcm(c.source)
    prepared = prepare(c)
    result = output(c, prepared.id)
    info = media.probe_trusted_media(result)
    assert abs(info.video_duration_sec - 895 / 30) < 1e-6
    assert picture_ids(result) == EXPECTED_OUTPUT_IDS
    decoded = preparation._decoded_picture(result, media._budget(30, None))
    assert decoded == [(Fraction(index, 1) / F, 1 / F) for index in range(895)]
    colours = final_colours(result)
    assert colours.shape == (5, 3)
    assert np.all(colours[:2, 1] > colours[:2, 2] + 70)
    assert np.all(colours[2:, 2] > colours[2:, 1] + 150)
    assert audio_identity(result) == sound
    assert pcm(result) == decoded_sound
    assert_audible_cta_tail(result)
    assert sha(c.source) == original_hash
    assert run_snapshot(c) == original_main
    with c.sessions() as session:
        artifact = get_version_artifact(session, prepared.id)
        original = get_version_artifact(session, c.original_id)
        assert original.sha256 == original_hash
        assert (prepared.parent_version_id, prepared.base_version_id, artifact.owner_version_id) == (c.original_id, c.original_id, prepared.id)
        roles = session.scalars(select(AssistantVersionArtifact.role).where(AssistantVersionArtifact.version_id == prepared.id)).all()
        assert not {"clean", "captions"}.intersection(roles)
        assert session.scalar(select(AssistantRunResultPromotion)) is None


def test_failed_audio_proof_removes_candidate_and_preserves_original(tmp_path, monkeypatch):
    source = gap_original(tmp_path)
    original_hash = sha(source)
    destination = tmp_path / "must-not-survive.mp4"
    real = preparation._audio_proof

    def changed(path, budget):
        proof = real(path, budget)
        return ("injected-encoded-audio-mismatch", proof) if path == destination else proof

    monkeypatch.setattr(preparation, "_audio_proof", changed)
    with pytest.raises(media.AssistantMediaError, match="AAC"):
        preparation.prepare_audio_tail(source, destination)
    assert not destination.exists()
    assert sha(source) == original_hash
    assert picture_ids(source, ignore_editlist=True) == list(range(N))


def test_nonuniform_video_bypass_is_refused_before_render(tmp_path, monkeypatch):
    source = gap_original(tmp_path)
    destination = tmp_path / "refused.mp4"
    real = preparation._video_packets

    def altered(path, budget, *, ignore_editlist=False):
        stream, packets = real(path, budget, ignore_editlist=ignore_editlist)
        if ignore_editlist:
            packets[10]["pts"] += 1
        return stream, packets

    monkeypatch.setattr(preparation, "_video_packets", altered)
    with pytest.raises(media.AssistantMediaError, match="recovery proof"):
        preparation.prepare_audio_tail(source, destination)
    assert not destination.exists()


def test_existing_pure_audio_tail_profile_remains_unchanged(tmp_path):
    source = native_original(tmp_path)
    with pytest.raises(media.EditingPreparationRequired) as error:
        media.probe_trusted_media(source)
    assert error.value.preparation["kind"] == "audio_tail"
    assert "gap_fill_frames" not in error.value.preparation
    destination = tmp_path / "ordinary-prepared.mp4"
    preparation.prepare_audio_tail(source, destination)
    assert abs(media.probe_trusted_media(destination).video_duration_sec - 895 / 30) < 1e-6
    assert audio_identity(destination) == audio_identity(source)
    assert pcm(destination) == pcm(source)


def test_strict_edited_output_publishes_and_archives_exact_grid_gap_original(grid_case):
    from app.assistant_run_result import history_media, promote_run_result, run_result_context
    c = grid_case
    original_hash, original_main = sha(c.source), run_snapshot(c)
    prepared = prepare(c)
    result = media.image_fades(output(c, prepared.id), c.data / "edited.mp4", fades=((29., 895 / 30, "out"),))
    edited_id = append_edit(c, prepared.id, result,
        {"kind": "image_fade", "start_sec": 29., "end_sec": 895 / 30, "direction": "out"}, "fade-after-grid-preparation")
    assert run_snapshot(c) == original_main
    assert_audible_cta_tail(result.output_path)
    drive = FakeDrive()
    with c.sessions() as session:
        receipt = promote_run_result(session, c.service, c.run_id, {"version_id": edited_id,
            "expected_selection_epoch": 0, "expected_result_revision": run_result_context(session, c.service, c.run_id)["revision"],
            "idempotency_key": "publish-owned-grid-output"}, drive=drive)
        archived = session.get(AssistantRunResultPromotion, receipt["id"])
        profile = archived.before_json["archive_profile"]
        assert profile["kind"] == "audio_tail_grid_gap_archive_only"
        assert sha(history_media(session, c.service, c.run_id, receipt["id"], "before")) == original_hash
        assert sha(history_media(session, c.service, c.run_id, receipt["id"], "after")) == sha(result.output_path)
        current = session.get(Run, c.run_id)
        assert current.result_gdrive_file_id == original_main["result_gdrive_file_id"]
        assert current.result_local_path != original_main["result_local_path"]
    assert drive.begins == drive.finishes == 1
    assert sha(c.source) == original_hash
