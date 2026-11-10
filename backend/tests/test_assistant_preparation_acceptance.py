"""Independent native acceptance for an original with an audible AAC overhang.

All media and SQLite state are owned synthetic fixtures. No production files,
external services, model calls or real Drive client are used.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app import assistant_media as media
from app.assistant_access import ProjectAllowlist
from app.assistant_access import AssistantAccessDenied
from app.assistant_legacy import import_legacy_run
from app.assistant_models import (
    AssistantArtifact, AssistantRunResultIntent, AssistantRunResultPromotion,
    AssistantSelection, AssistantVersion, AssistantVersionArtifact,
)
from app.assistant_store import (
    ArtifactInput, AssistantArtifactStorage, AssistantConflict, get_version_artifact, publish_version,
    reserve_version, select_version, verify_artifact,
)
from app.assistant_contracts import EditPlan
from app.assistant_turns import AssistantTurnService
from app.db import Base
from app.models import Run, VideoProject

VIDEO_FRAMES = 892
FPS = Fraction(30)
RATE = 44100
AUDIO_SAMPLES = 1315106
VIDEO_END = float(Fraction(VIDEO_FRAMES, 1) / FPS)
PREPARED_END = float(Fraction(VIDEO_FRAMES + 3, 1) / FPS)


def command(argv):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(path, *options):
    return json.loads(command(["ffprobe", "-v", "error", *options, "-of", "json", str(path)]))


def audio_identity(path):
    """Compare the encoded sound and its native presentation, not container offsets."""
    stream = probe(path, "-select_streams", "a:0", "-show_streams", "-show_data_hash", "sha256")["streams"][0]
    clock = Fraction(stream["time_base"])
    packets = probe(path, "-select_streams", "a:0", "-show_packets", "-show_data_hash", "sha256")["packets"]
    timeline = [(
        Fraction(item["pts"]) * clock, Fraction(item["dts"]) * clock,
        Fraction(item["duration"]) * clock, item["size"], item["data_hash"],
        item.get("side_data_list", []),
    ) for item in packets]
    shape = tuple(stream.get(key) for key in ("codec_name", "sample_rate", "channels", "channel_layout", "extradata_hash"))
    extent = (Fraction(stream.get("start_pts", 0)) * clock, Fraction(stream["duration_ts"]) * clock)
    return shape, extent, timeline


def pcm(path):
    return command(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
                    "-c:a", "pcm_s16le", "-f", "s16le", "-"])


def assert_audible_cta_tail(path, *, shifted=0):
    samples = np.frombuffer(pcm(path), dtype="<i2").reshape(-1, 2)[:, 0].astype(float) / 32768
    first = round((VIDEO_END + .015 - shifted) * RATE)
    last = round((float(Fraction(AUDIO_SAMPLES, RATE)) - .008 - shifted) * RATE)
    tail = samples[first:last]
    assert len(tail) > 2000, "the audio after the original final frame was truncated"
    assert np.sqrt(np.mean(tail ** 2)) > .05, "audible CTA tail became silence"
    frequencies = np.fft.rfftfreq(len(tail), 1 / RATE)
    peak = frequencies[np.argmax(abs(np.fft.rfft(tail)))]
    assert abs(peak - 1320) < 35, f"wrong audio at the CTA tail: {peak:g} Hz"


def video_frame_means(path):
    raw = command(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                   "-vf", "select=gte(n\\,890),format=rgb24", "-vsync", "0", "-f", "rawvideo", "-"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 120, 160, 3).mean(axis=(1, 2))


def native_original(data):
    video, audio, source = data / "picture.mp4", data / "sound.m4a", data / "original.mp4"
    # Separate finite streams avoid -shortest or a frame limit clipping AAC.
    command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
        "color=c=blue:s=160x120:r=30", "-frames:v", str(VIDEO_FRAMES),
        "-vf", "drawbox=c=red:t=fill:enable='eq(n,891)'", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-threads", "1", "-pix_fmt", "yuv420p", str(video)])
    # The distinguishing tone begins just before picture end and remains
    # audible in the entire 87.665 ms overhang, rather than merely AAC silence.
    signal = "0.2*sin(2*PI*if(lt(t,29.70),440,1320)*t)"
    command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
        f"aevalsrc='{signal}':s={RATE}", "-af", f"atrim=end_sample={AUDIO_SAMPLES},asetpts=N/SR/TB",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE), "-ac", "2",
        "-movie_timescale", str(RATE), str(audio)])
    command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-movie_timescale", str(RATE),
        "-movflags", "+faststart", str(source)])
    return source


@pytest.fixture
def incident(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = native_original(data)
    engine = create_engine(f"sqlite:///{tmp_path / 'isolated.db'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    storage = AssistantArtifactStorage(data)
    with sessions.begin() as session:
        project = VideoProject(source_type="upload", source_ref="synthetic-cta.mp4", status="ready")
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status="done", prompt="unchanged operator recipe",
                  result_local_path=str(source), result_duration_sec=float(Fraction(AUDIO_SAMPLES, RATE)),
                  result_gdrive_file_id="existing-drive-id")
        session.add(run)
        session.flush()
        run_id, project_id = run.id, project.id
    service = AssistantTurnService(storage=storage, policy=ProjectAllowlist(project_id), enabled=True)
    with sessions.begin() as session:
        original = import_legacy_run(session, session.get(Run, run_id), actor="operator", storage=storage, policy=service.policy)
        original_id, conversation_id = original.id, original.conversation_id
    yield SimpleNamespace(data=data, source=source, sessions=sessions, storage=storage,
        run_id=run_id, project_id=project_id, service=service, original_id=original_id,
        conversation_id=conversation_id)
    engine.dispose()


def request(c, *, key="prepare-native-tail", version_id=None, epoch=0):
    return {"version_id": version_id or c.original_id, "expected_selection_epoch": epoch, "idempotency_key": key}


def run_snapshot(c):
    with c.sessions() as session:
        row = session.get(Run, c.run_id)
        return {column.name: getattr(row, column.name) for column in Run.__table__.columns}


def output(c, version_id):
    with c.sessions() as session:
        return verify_artifact(get_version_artifact(session, version_id, role="full"), storage=c.storage)


def prepare(c, body=None, *, run_id=None):
    from app.assistant_preparation import prepare_editable_version
    body = request(c) if body is None else body
    with c.sessions() as session:
        return prepare_editable_version(session, c.service, run_id or c.run_id, body["version_id"],
            expected_selection_epoch=body["expected_selection_epoch"], idempotency_key=body["idempotency_key"])


def append_edit(c, base_id, result, operation, key):
    plan = EditPlan.model_validate({"run_id": c.run_id, "base_version_id": base_id, "operations": [operation]})
    with c.sessions.begin() as session:
        child = reserve_version(session, plan, idempotency_key=key, actor="operator")
        item = ArtifactInput(result.output_path, metadata={"duration_sec": result.metadata.video_duration_sec,
            "has_audio": result.metadata.has_audio, "fps": result.metadata.fps, "subtitle_status": "unknown"})
        publish_version(session, child.id, full=item, preview=item, storage=c.storage)
        return child.id


class FakeDrive:
    def __init__(self):
        self.begins = self.finishes = 0
        self.content = None

    def begin_result_update(self, file_id, path, *, deadline):
        assert file_id == "existing-drive-id"
        self.begins += 1
        return "https://www.googleapis.com/upload/drive/v3/files/fixture?upload_id=owned-test"

    def finish_result_update(self, uri, file_id, path, **kwargs):
        assert file_id == "existing-drive-id"
        self.finishes += 1
        self.content = Path(path).read_bytes()
        return {"id": file_id}


def test_owned_fixture_reproduces_picture_clock_and_audible_aac_overhang(tmp_path):
    source = native_original(tmp_path)
    streams = probe(source, "-show_streams")["streams"]
    video = next(item for item in streams if item["codec_type"] == "video")
    audio = next(item for item in streams if item["codec_type"] == "audio")
    assert (video["codec_name"], video["avg_frame_rate"], int(video["nb_frames"])) == ("h264", "30/1", VIDEO_FRAMES)
    assert (audio["codec_name"], int(audio["sample_rate"]), int(audio["channels"])) == ("aac", RATE, 2)
    assert float(audio["start_time"]) == 0
    assert .085 < float(audio["duration"]) - float(video["duration"]) < .090
    with pytest.raises(media.AssistantMediaError):
        media.probe_trusted_media(source)
    assert_audible_cta_tail(source)


def test_explicit_preparation_preserves_complete_audio_and_original_state(incident):
    c = incident
    before_run, before_sha = run_snapshot(c), sha(c.source)
    before_audio, before_pcm = audio_identity(c.source), pcm(c.source)
    with c.sessions() as session:
        original = session.get(AssistantVersion, c.original_id)
        frozen_recipe = json.dumps(original.recipe, sort_keys=True)
        original_artifact = get_version_artifact(session, original.id)
        assert original_artifact.sha256 == before_sha
        assert original_artifact.metadata_json["needs_editing_preparation"] is True
        original_artifact_id = original_artifact.id
    prepared = prepare(c)
    prepared_path = output(c, prepared.id)
    info = media.probe_trusted_media(prepared_path)
    assert abs(info.video_duration_sec - PREPARED_END) < 1e-6
    assert (info.fps, info.width, info.height, info.sample_aspect_ratio, info.has_audio) == ("30", 160, 120, "1:1", True)
    frame_count = probe(prepared_path, "-select_streams", "v:0", "-count_frames", "-show_streams")["streams"][0]["nb_read_frames"]
    assert int(frame_count) == VIDEO_FRAMES + 3
    colors = video_frame_means(prepared_path)
    assert colors.shape == (5, 3)
    assert colors[0, 2] > colors[0, 0] + 100
    assert np.all(colors[1:, 0] > colors[1:, 2] + 100), "the last original red frame must be held for all three new frames"
    assert audio_identity(prepared_path) == before_audio
    assert pcm(prepared_path) == before_pcm
    assert_audible_cta_tail(prepared_path)
    assert run_snapshot(c) == before_run
    assert sha(c.source) == before_sha
    with c.sessions() as session:
        original = session.get(AssistantVersion, c.original_id)
        current = session.get(AssistantVersion, prepared.id)
        assert json.dumps(original.recipe, sort_keys=True) == frozen_recipe
        assert get_version_artifact(session, original.id).id == original_artifact_id
        assert get_version_artifact(session, original.id).sha256 == before_sha
        assert (current.parent_version_id, current.base_version_id) == (c.original_id, c.original_id)
        assert get_version_artifact(session, prepared.id).owner_version_id == prepared.id
        roles = session.scalars(select(AssistantVersionArtifact.role).where(AssistantVersionArtifact.version_id == prepared.id)).all()
        assert {"full", "preview"} <= set(roles)
        assert not {"clean", "captions"}.intersection(roles), "preparation cannot invent clean picture or editable subtitles"
        assert session.scalar(select(AssistantRunResultPromotion)) is None
        assert session.scalar(select(AssistantRunResultIntent)) is None


def test_duplicate_preparation_replays_without_reselecting_or_rendering(incident, monkeypatch):
    from app import assistant_preparation
    c = incident
    prepared = prepare(c)
    prepared_sha = sha(output(c, prepared.id))
    with c.sessions.begin() as session:
        selected = select_version(session, c.conversation_id, c.original_id, expected_epoch=0)
        selected_epoch = selected.epoch
        counts = tuple(len(session.scalars(select(model)).all()) for model in (AssistantVersion, AssistantArtifact, AssistantVersionArtifact))

    def forbidden(*args, **kwargs):
        pytest.fail("an idempotent preparation replay must not re-render")

    monkeypatch.setattr(assistant_preparation, "prepare_audio_tail", forbidden)
    assert prepare(c).id == prepared.id
    with c.sessions() as session:
        selected = session.get(AssistantSelection, c.conversation_id)
        assert (selected.selected_version_id, selected.epoch) == (c.original_id, selected_epoch)
        assert tuple(len(session.scalars(select(model)).all()) for model in (AssistantVersion, AssistantArtifact, AssistantVersionArtifact)) == counts
    assert sha(output(c, prepared.id)) == prepared_sha
    # A new request can find the already-prepared copy without selecting it.
    assert prepare(c, request(c, key="find-existing-copy", epoch=selected_epoch)).id == prepared.id
    with c.sessions() as session:
        assert session.get(AssistantSelection, c.conversation_id).selected_version_id == c.original_id


@pytest.mark.parametrize("reason", ["stale_epoch", "wrong_run", "unresolved_publication"])
def test_preparation_refuses_stale_or_wrong_owner_without_new_version(incident, reason):
    c = incident
    before = run_snapshot(c)
    body = request(c)
    run_id = c.run_id
    if reason == "stale_epoch":
        body["expected_selection_epoch"] = 1
    elif reason == "wrong_run":
        with c.sessions.begin() as session:
            sibling = Run(project_id=c.project_id, status="done", result_local_path=str(c.source), result_duration_sec=VIDEO_END)
            session.add(sibling)
            session.flush()
            run_id = sibling.id
            import_legacy_run(session, sibling, actor="operator", storage=c.storage, policy=c.service.policy)
    else:
        with c.sessions.begin() as session:
            artifact = get_version_artifact(session, c.original_id)
            session.add(AssistantRunResultIntent(run_id=c.run_id, conversation_id=c.conversation_id,
                version_id=c.original_id, artifact_id=artifact.id, idempotency_key="owned-unresolved-intent",
                request_sha256="a" * 64, actor="operator", request_json={}, before_json={}, after_json={}))
    with c.sessions() as session:
        count_before = len(session.scalars(select(AssistantVersion)).all())
    with pytest.raises((AssistantConflict, AssistantAccessDenied)):
        prepare(c, body, run_id=run_id)
    assert run_snapshot(c) == before
    with c.sessions() as session:
        assert len(session.scalars(select(AssistantVersion)).all()) == count_before
        assert session.get(AssistantSelection, c.conversation_id).selected_version_id == c.original_id


@pytest.mark.parametrize("operation", ["fade", "trim", "cut"])
def test_prepared_picture_edits_preserve_audible_final_cta(incident, operation):
    c = incident
    prepared = prepare(c)
    source = output(c, prepared.id)
    destination = c.data / f"{operation}.mp4"
    if operation == "fade":
        result = media.image_fades(source, destination, fades=((29., PREPARED_END, "out"),))
        assert audio_identity(destination) == audio_identity(source)
        shifted = 0
    elif operation == "trim":
        result = media.trim_video(source, destination, start_sec=1, end_sec=PREPARED_END)
        shifted = 1
    else:
        result = media.cut_section(source, destination, start_sec=1, end_sec=2)
        shifted = 1
    assert abs(result.metadata.video_duration_sec - (PREPARED_END - shifted)) < 1e-6
    assert_audible_cta_tail(destination, shifted=shifted)
    assert sha(c.source) == get_original_sha(c)


def get_original_sha(c):
    with c.sessions() as session:
        return get_version_artifact(session, c.original_id).sha256


def test_prepared_fade_and_cut_can_publish_while_archiving_exact_original(incident):
    from app.assistant_run_result import history_media, promote_run_result, run_result_context
    c = incident
    original_sha, original_run = sha(c.source), run_snapshot(c)
    prepared = prepare(c)
    source = output(c, prepared.id)
    prepared_sha = sha(source)
    fade = media.image_fades(source, c.data / "fade-child.mp4", fades=((29., PREPARED_END, "out"),))
    fade_id = append_edit(c, prepared.id, fade,
        {"kind": "image_fade", "start_sec": 29., "end_sec": PREPARED_END, "direction": "out"}, "native-fade")
    cut = media.cut_section(output(c, fade_id), c.data / "cut-child.mp4", start_sec=1, end_sec=2)
    cut_id = append_edit(c, fade_id, cut, {"kind": "cut", "start_sec": 1, "end_sec": 2}, "native-cut")
    assert run_snapshot(c) == original_run, "preparation and creative versions must not change the main result"
    assert_audible_cta_tail(cut.output_path, shifted=1)
    drive = FakeDrive()
    with c.sessions() as session:
        selection = session.get(AssistantSelection, c.conversation_id)
        body = {"version_id": cut_id, "expected_selection_epoch": selection.epoch,
            "expected_result_revision": run_result_context(session, c.service, c.run_id)["revision"],
            "idempotency_key": "publish-prepared-cut"}
        receipt = promote_run_result(session, c.service, c.run_id, body, drive=drive)
        current = session.get(Run, c.run_id)
        assert current.result_gdrive_file_id == original_run["result_gdrive_file_id"]
        assert current.final_creatives_exported_at == original_run["final_creatives_exported_at"]
        assert abs(current.result_duration_sec - (PREPARED_END - 1)) < 1e-6
        assert Path(current.result_local_path).read_bytes() == Path(cut.output_path).read_bytes() == drive.content
        assert sha(history_media(session, c.service, c.run_id, receipt["id"], "before")) == original_sha
        assert sha(history_media(session, c.service, c.run_id, receipt["id"], "after")) == sha(cut.output_path)
        assert get_version_artifact(session, c.original_id).sha256 == original_sha
        assert sha(output(c, prepared.id)) == prepared_sha
    assert drive.begins == drive.finishes == 1
    assert sha(c.source) == original_sha
