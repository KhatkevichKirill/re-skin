"""AICP-345: the AICP-343 QA tools behind the assistant's closed tool boundary.

Real confined runs (the native media launcher) on small synthetic clips: the
confined backend must decode exactly what the CLI backend decodes, and the
agent media service must return bounded, path-free evidence bound to exact
authorized inputs.
"""
from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np
import pytest
from pydantic import TypeAdapter
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import assistant_media as amedia
from app import qa_tools as q
from app.assistant_access import ProjectAllowlist
from app.assistant_agent_contracts import (AgentObservation, AgentStepContext, CheckAlignmentAction,
    CompareFramesAction, DetectCaptionsAction, QAEvidence, validate_agent_action)
from app.assistant_agent_media import AgentMediaService, AgentToolUnavailable
from app.assistant_qa import ConfinedQAMedia
from app.assistant_store import AssistantArtifactStorage, get_version_artifact, import_run_version
from app.db import Base
from app.models import Run, VideoProject

W, H, FPS, SECONDS = 270, 480, 30, 3


def _caption_frame(i, text):
    img = np.full((H, W, 3), 128, np.uint8)
    # A moving block so alignment has something to lock on to.
    x = (i * 5) % (W - 40)
    img[40:80, x:x + 40] = (30, 160, 220)
    if text:
        mask = np.zeros((H, W), np.uint8)
        cv2.putText(mask, "WORD", (int(W * 0.25), int(H * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 3, cv2.LINE_8)
        outline = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        img[outline > 0] = 0
        img[mask > 0] = 255
    return img


def _encode(path, *, captions):
    frames = [_caption_frame(i, captions and 15 <= i < 75) for i in range(SECONDS * FPS)]
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={SECONDS}",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "1", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        input=b"".join(f.tobytes() for f in frames), capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    return path


@pytest.fixture
def qa(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'qa.db'}")

    @event.listens_for(engine, "connect")
    def pragma(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    data = tmp_path / "data"
    data.mkdir()
    storage = AssistantArtifactStorage(data)
    source = _encode(data / "source.mp4", captions=True)
    clean = _encode(data / "final.mp4", captions=False)
    with factory.begin() as session:
        project = VideoProject(source_type="upload", source_ref="fixture", source_local_path=str(source))
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status="done", result_local_path=str(clean), result_duration_sec=SECONDS)
        session.add(run)
        session.flush()
        run_id, project_id = run.id, project.id
    with factory.begin() as session:
        version = import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage)
        version_id = version.id
        asset_id = get_version_artifact(session, version_id).id
    with AgentMediaService(factory, storage=storage, policy=ProjectAllowlist([project_id]),
                           target_version_id=version_id) as service:
        original = service.search_sources(query="original")["sources"][0]
        yield {"service": service, "factory": factory, "project_id": project_id, "run_id": run_id,
               "selected": {"asset_id": asset_id, "version_id": version_id},
               "original": {"asset_id": original["asset_id"], "version_id": original["version_id"]},
               "source": str(source), "clean": str(clean)}
    engine.dispose()


def _confined():
    return ConfinedQAMedia(remaining=lambda: 120, check=lambda: False)


def test_confined_backend_decodes_exactly_what_the_cli_decodes(qa):
    local, confined = q.LocalMedia(), _confined()
    for kwargs in ({"fps": 10, "width": 270, "height": 480},
                   {"fps": 30, "width": 72, "height": 128, "pix_fmt": "gray", "start": 0.5, "duration": 1.2}):
        a = list(local.frames(qa["source"], **kwargs))
        b = list(confined.frames(qa["source"], **kwargs))
        assert len(a) == len(b) > 0 and all(np.array_equal(x, y) for x, y in zip(a, b))
    for t in (0.0, 1.27, 2.9):
        assert np.array_equal(local.grab(qa["source"], t), confined.grab(qa["source"], t))
    assert confined.grab(qa["source"], 5.0) is None
    assert confined.probe(qa["source"]).duration_sec == pytest.approx(local.probe(qa["source"]).duration_sec)
    with q.use_media(confined):
        scan = q.detect_captions(qa["clean"], calibrate=qa["source"])
    assert [(f.t, f.score) for f in scan.scores] == [(f.t, f.score) for f in q.detect_captions(
        qa["clean"], calibrate=qa["source"]).scores]


def test_stream_is_bounded_and_reaped(qa):
    budget = amedia._budget(30, None)
    argv = q._decode_argv(qa["source"], fps=30, width=72, height=128, pix_fmt="gray", start=None, duration=None)
    with pytest.raises(amedia.AssistantMediaError, match="record limit"):
        list(amedia._stream(argv, budget, inputs=(qa["source"],), chunk_bytes=72 * 128, max_chunks=5))
    stream = amedia._stream(argv, budget, inputs=(qa["source"],), chunk_bytes=72 * 128, max_chunks=500)
    assert len(next(stream)) == 72 * 128
    stream.close()  # abandoning the stream kills and reaps the decoder
    with pytest.raises(amedia.AssistantMediaError):
        list(amedia._stream(argv, budget, inputs=(qa["source"],), chunk_bytes=0, max_chunks=1))
    cancelled = amedia._budget(30, lambda: True)
    with pytest.raises(amedia.AssistantMediaCancelled):
        list(amedia._stream(argv, cancelled, inputs=(qa["source"],), chunk_bytes=72 * 128, max_chunks=500))


def test_detect_captions_calibrated_on_the_original(qa):
    service = qa["service"]
    clean = service.detect_captions(target=qa["selected"], calibration=qa["original"])
    evidence = TypeAdapter(QAEvidence).validate_python(clean)
    assert evidence.verdict == "clean" and evidence.hits == 0 and evidence.frames == 30
    assert evidence.calibration.calibrated and evidence.calibration.hits > 0
    assert evidence.target.asset_id == qa["selected"]["asset_id"] and not evidence.project_config
    dirty = service.detect_captions(target=qa["original"], calibration=qa["original"], start_sec=1.0, duration_sec=1.5)
    assert dirty["verdict"] == "captions_found" and dirty["spans"] and dirty["spans_total"] == len(dirty["spans"])
    assert "/" not in json.dumps(clean)  # no paths leave the boundary


def test_calibration_must_be_the_imported_original(qa):
    with pytest.raises(AgentToolUnavailable, match="Original project source"):
        qa["service"].detect_captions(target=qa["original"], calibration=qa["selected"])


def test_project_detector_config_is_applied(qa):
    with qa["factory"].begin() as session:
        session.get(VideoProject, qa["project_id"]).caption_detector = {"text_color": "#ffff00"}
    out = qa["service"].detect_captions(target=qa["selected"], calibration=qa["original"])
    assert out["project_config"] and out["verdict"] == "clean_uncalibrated"
    with qa["factory"].begin() as session:
        session.get(VideoProject, qa["project_id"]).caption_detector = {"threshold": "nope"}
    with pytest.raises(AgentToolUnavailable, match="config is invalid"):
        qa["service"].detect_captions(target=qa["selected"], calibration=qa["original"])


def test_alignment_and_comparison(qa):
    service = qa["service"]
    out = service.check_alignment(source=qa["original"], result=qa["selected"])
    evidence = TypeAdapter(QAEvidence).validate_python(out)
    assert evidence.mode == "timeline" and evidence.length_ok and evidence.points > 0
    window = service.check_alignment(source=qa["original"], result=qa["selected"], mode="window",
                                     source_start_sec=0.0, duration_sec=3.0)
    assert TypeAdapter(QAEvidence).validate_python(window).samples
    compare = service.compare_frames(source=qa["original"], result=qa["selected"],
        windows=[{"source_start_sec": 0.0, "source_end_sec": 3.0, "label": "all"}], every_sec=0.5)
    assert len(compare["samples"]) == 6 and all(s["diff"] is not None for s in compare["samples"])
    # captions present only in the source make the middle pairs differ more than the edges
    diffs = [s["diff"] for s in compare["samples"]]
    assert max(diffs[1:4]) > diffs[-1]
    with pytest.raises(AgentToolUnavailable, match="48 frame pairs"):
        service.compare_frames(source=qa["original"], result=qa["selected"],
            windows=[{"source_start_sec": 0.0, "source_end_sec": 3.0}] * 8, every_sec=0.25)


def test_qa_rechecks_authorization(qa):
    from app.assistant_access import AssistantAccessDenied
    qa["service"].policy = ProjectAllowlist([])
    with pytest.raises(AssistantAccessDenied):
        qa["service"].detect_captions(target=qa["selected"], calibration=qa["original"])


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def _ids():
    import uuid
    return [str(uuid.uuid4()) for _ in range(4)]


def test_qa_actions_reject_ambiguous_or_misplaced_inputs():
    run, base, asset, version = _ids()
    scope = {"run_id": run, "base_version_id": base}
    ref = {"asset_id": asset, "version_id": version}
    with pytest.raises(ValueError):
        DetectCaptionsAction(kind="detect_captions", **scope, target=ref, calibration={"candidate_id": asset})
    with pytest.raises(ValueError):
        DetectCaptionsAction(kind="detect_captions", **scope, target={**ref, "candidate_id": asset}, calibration=ref)
    with pytest.raises(ValueError):
        CheckAlignmentAction(kind="check_alignment", **scope, source=ref, result=ref, mode="window", source_start_sec=0.0)
    with pytest.raises(ValueError):
        CheckAlignmentAction(kind="check_alignment", **scope, source=ref, result=ref, source_start_sec=1.0)
    with pytest.raises(ValueError):
        CompareFramesAction(kind="compare_frames", **scope, source=ref, result=ref,
                            windows=[{"source_start_sec": 2.0, "source_end_sec": 1.0}])
    with pytest.raises(ValueError):
        CompareFramesAction(kind="compare_frames", **scope, source=ref, result=ref, windows=[])
    with pytest.raises(ValueError):  # no paths: unknown fields are refused
        DetectCaptionsAction(kind="detect_captions", **scope, target={**ref, "path": "/etc/passwd"}, calibration=ref)


def _evidence(asset, version, sha="a" * 64, spans=0):
    ref = {"asset_id": asset, "version_id": version, "sha256": sha}
    return {"tool": "detect_captions", "target": ref, "config_sha256": sha, "project_config": False, "fps": 10.0,
            "threshold": 40, "frames": 30, "hits": 0, "hit_ratio": 0.0, "verdict": "clean",
            "spans": [{"start_sec": float(i), "end_sec": float(i), "frames": 1, "max_score": 50} for i in range(spans)],
            "spans_total": spans, "scores_sha256": sha,
            "calibration": {"source": ref, "frames": 30, "hits": 10, "hit_ratio": 0.33, "calibrated": True}}


def test_qa_observation_payload_is_exact_and_bounded():
    _, _, asset, version = _ids()
    obs = AgentObservation(kind="qa", action_index=0, qa=_evidence(asset, version))
    assert obs.qa.verdict == "clean"
    with pytest.raises(ValueError):
        AgentObservation(kind="frames", action_index=0, qa=_evidence(asset, version))
    with pytest.raises(ValueError):
        AgentObservation(kind="qa", action_index=0)
    with pytest.raises(ValueError):  # more spans than the evidence carries
        AgentObservation(kind="qa", action_index=0, qa=_evidence(asset, version, spans=25))
    with pytest.raises(ValueError):  # a path is not a field
        AgentObservation(kind="qa", action_index=0, qa={**_evidence(asset, version), "path": "/x"})


def test_qa_actions_and_evidence_bind_to_the_pinned_context():
    from app.assistant_agent_contracts import AgentActionEnvelope
    from tests.test_assistant_planner import context, RUN, BASE, ASSET, OTHER
    ctx = context()
    ref = {"asset_id": ASSET, "version_id": BASE}

    def act(**extra):
        return AgentActionEnvelope.model_validate({"action": {"run_id": RUN, "base_version_id": BASE, **extra}}).action

    step = AgentStepContext(context=ctx, step_index=0)
    ok = act(kind="detect_captions", target=ref, calibration=ref)
    assert validate_agent_action(ok, step).kind == "detect_captions"
    for bad in (act(kind="detect_captions", target={"asset_id": OTHER, "version_id": BASE}, calibration=ref),
                act(kind="detect_captions", target={"candidate_id": OTHER}, calibration=ref),
                act(kind="detect_captions", target=ref, calibration=ref, start_sec=ctx.selected.duration_sec),
                act(kind="check_alignment", source=ref, result=ref, mode="window", source_start_sec=0.0,
                    duration_sec=ctx.selected.duration_sec + 1) if ctx.selected.duration_sec < 29 else None,
                act(kind="compare_frames", source=ref, result=ref,
                    windows=[{"source_start_sec": 0.0, "source_end_sec": ctx.selected.duration_sec + 1}])):
        if bad is None:
            continue
        with pytest.raises(ValueError):
            validate_agent_action(bad, step)
    good = AgentObservation(kind="qa", action_index=0, qa=_evidence(ASSET, BASE))
    AgentStepContext(context=ctx, step_index=1, observations=(good,))
    with pytest.raises(ValueError, match="authorized video"):
        AgentStepContext(context=ctx, step_index=1, observations=(
            AgentObservation(kind="qa", action_index=0, qa=_evidence(OTHER, BASE)),))
    with pytest.raises(ValueError, match="prior candidate"):
        AgentStepContext(context=ctx, step_index=1, observations=(AgentObservation(kind="qa", action_index=0,
            qa={**_evidence(ASSET, BASE), "target": {"candidate_id": OTHER, "sha256": "a" * 64}}),))


def test_qa_history_byte_budget():
    from tests.test_assistant_planner import context, BASE, ASSET
    big = _evidence(ASSET, BASE, spans=24)
    one = AgentObservation(kind="qa", action_index=0, qa=big).qa
    size = len(json.dumps(one.model_dump(mode="json"), ensure_ascii=False).encode())
    from app.assistant_agent_contracts import MAX_QA_HISTORY_BYTES
    count = MAX_QA_HISTORY_BYTES // size + 1
    assert count <= 11, size
    observations = tuple(AgentObservation(kind="qa", action_index=i, qa=big) for i in range(count))
    with pytest.raises(ValueError, match="QA history"):
        AgentStepContext(context=context(), step_index=count, observations=observations)
    AgentStepContext(context=context(), step_index=count - 1, observations=observations[:count - 1])


def test_script_captions_through_the_confined_service(qa):
    import uuid as _uuid
    from app.models import RunSegment
    service = qa["service"]
    with pytest.raises(AgentToolUnavailable, match="snapshots"):
        service.script_captions()
    definition = str(_uuid.uuid4())
    with qa["factory"].begin() as session:
        run = session.get(Run, qa["run_id"])
        run.segment_snapshot = [{"id": definition, "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 3.0}]
        run.audio_mode = "seedance"
        session.add(RunSegment(run_id=run.id, segment_def_id=definition, index=0, status="completed",
                               local_result_path=qa["clean"], prompt_override="[0.0–1.5] A: Hello there."))
    out = service.script_captions()
    evidence = TypeAdapter(QAEvidence).validate_python(out)
    assert [line.text for line in evidence.lines] == ["Hello there."] and evidence.lineage == "run_segments"
    # The run result moved on (e.g. an audio patch): the selected version no longer matches.
    with qa["factory"].begin() as session:
        session.get(Run, qa["run_id"]).result_local_path = qa["source"]
        session.get(RunSegment, session.query(RunSegment).filter_by(run_id=qa["run_id"]).one().id).local_result_path = qa["source"]
    with pytest.raises(AgentToolUnavailable, match="not the run's current result"):
        service.script_captions()
