"""AICP-345: captions from the script (resolver) and the operator's Apply."""
from __future__ import annotations

import hashlib
import subprocess
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app import assistant_script_captions as script
from app import qa_tools as q
from app.assistant_qa import digest
from app.db import Base
from app.models import CaptionStatus, LibraryClip, Run, RunCaption, RunSegment, VideoProject


def _clip(path, seconds):
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"testsrc2=duration={seconds}:size=160x120:rate=24", "-f", "lavfi", "-i",
        f"sine=frequency=300:sample_rate=48000:duration={seconds}", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    return str(path)


def _concat(paths, out):
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p}'\n" for p in paths))
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                           "-c", "copy", str(out)], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    return str(out)


PROMPTS = ["Header.\n[0.0–1.0] A: First line.\n[1.0–2.0] B: Second line.",
           "Header.\n[0.2–1.4] A: Third line."]


@pytest.fixture
def world(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'s.db'}")

    @event.listens_for(engine, "connect")
    def pragma(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    data = tmp_path / "data"
    data.mkdir()
    clips = [_clip(data / "r0.mp4", 2), _clip(data / "r1.mp4", 1.5)]
    final = _concat(clips, data / "final.mp4")
    defs = [str(uuid.uuid4()), str(uuid.uuid4())]
    snapshot = [{"id": defs[0], "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 2.0},
                {"id": defs[1], "index": 1, "action": "swap", "start_sec": 2.0, "end_sec": 3.5}]
    with factory.begin() as session:
        project = VideoProject(source_type="upload", source_ref="fixture")
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status="done", result_local_path=final, audio_mode="seedance",
                  segment_snapshot=snapshot, prompt="")
        session.add(run)
        session.flush()
        segments = []
        for i, (d, clip, prompt) in enumerate(zip(defs, clips, PROMPTS)):
            rs = RunSegment(run_id=run.id, segment_def_id=d, index=i, status="completed",
                            local_result_path=clip, prompt_override=prompt)
            session.add(rs)
            segments.append(rs)
        caption = RunCaption(run_id=run.id, status=CaptionStatus.done, source_type="run",
                             lines=[{"start": 0.0, "end": 1.0, "text": "old"}], lines_source="transcribe")
        session.add(caption)
        session.flush()
        ids = SimpleNamespace(project=project.id, run=run.id, defs=defs, segments=[s.id for s in segments],
                              caption=caption.id, clips=clips, final=final, data=data.resolve())
    yield factory, ids
    engine.dispose()


def _resolve(factory, ids, donors=None):
    with factory() as session:
        return script.resolve(session, run=session.get(Run, ids.run), library_donors=donors or {},
                              data_root=ids.data, probe=q.LocalMedia().probe, check=lambda: None)


def test_lines_follow_the_stored_prompts_and_real_clip_lengths(world):
    factory, ids = world
    out = _resolve(factory, ids)
    expected = q.captions_from_script([
        {"source_start": 0.0, "source_end": 2.0, "clip_duration": q.LocalMedia().probe(ids.clips[0]).duration_sec,
         "prompt": PROMPTS[0]},
        {"source_start": 2.0, "source_end": 3.5, "clip_duration": q.LocalMedia().probe(ids.clips[1]).duration_sec,
         "prompt": PROMPTS[1]}])
    assert out["lines"] == expected and len(out["lines"]) == 3
    assert out["lineage"] == "run_segments" and out["warnings"] == []
    assert out["final_sha256"] == hashlib.sha256(open(ids.final, "rb").read()).hexdigest()
    assert out["pieces"][1]["offset_sec"] == pytest.approx(out["pieces"][0]["clip_duration_sec"], abs=1e-3)
    target = out["captions"][0]
    assert target["caption_id"] == ids.caption and target["apply_allowed"]
    assert target["lines_sha256"] == digest([{"start": 0.0, "end": 1.0, "text": "old"}])
    from app.assistant_agent_contracts import ScriptCaptionsEvidence
    ScriptCaptionsEvidence.model_validate(out)


@pytest.mark.parametrize("change, message", [
    (lambda run, s: setattr(run, "audio_mode", "original"), "original soundtrack"),
    (lambda run, s: setattr(run, "stitch_plan", []), "no-stitch"),
    (lambda run, s: setattr(run, "segment_snapshot", None), "snapshots"),
    (lambda run, s: setattr(run, "segment_snapshot", [{**run.segment_snapshot[0], "action": "keep"},
                                                       run.segment_snapshot[1]]), "keep pieces"),
    (lambda run, s: setattr(s, "source_fallback", True), "completed generated clip"),
    (lambda run, s: setattr(run, "result_local_path", s.local_result_path), "length does not match"),
])
def test_refusals_fail_closed(world, change, message):
    factory, ids = world
    with factory.begin() as session:
        run = session.get(Run, ids.run)
        change(run, session.get(RunSegment, ids.segments[0]))
    with pytest.raises(script.ScriptCaptionsRefused, match=message):
        _resolve(factory, ids)


def test_library_entry_needs_a_matching_donor(world):
    factory, ids = world
    # A sibling run of the same project generated segment 0; its clip went
    # to the library and this run's plan uses it instead of its own def 0.
    with factory.begin() as session:
        run = session.get(Run, ids.run)
        donor_run = Run(project_id=ids.project, status="done", segment_snapshot=run.segment_snapshot, prompt="")
        session.add(donor_run)
        session.flush()
        donor = RunSegment(run_id=donor_run.id, segment_def_id=ids.defs[0], index=0, status="completed",
                           local_result_path=ids.clips[0], prompt_override=PROMPTS[0])
        session.add(donor)
        session.delete(session.get(RunSegment, ids.segments[0]))
        library = LibraryClip(name="TEMP", local_path=ids.clips[0])
        session.add(library)
        session.flush()
        run.stitch_plan = [{"kind": "library", "clip_id": library.id},
                           {"kind": "segment", "segment_def_id": ids.defs[1]}]
        donor_id, library_id, donor_run_id = donor.id, library.id, donor_run.id
    with pytest.raises(script.ScriptCaptionsRefused, match="needs the run segment"):
        _resolve(factory, ids)
    out = _resolve(factory, ids, {0: donor_id})
    assert out["lineage"] == "library_donor_attestation_required" and out["warnings"] == []
    assert out["lines"] == _baseline_lines(ids)
    # A donor from another window, a mismatching library file, a deleted library row.
    with factory.begin() as session:
        session.get(Run, donor_run_id).segment_snapshot = [{**session.get(Run, ids.run).segment_snapshot[0], "end_sec": 1.9}]
    with pytest.raises(script.ScriptCaptionsRefused, match="replaced swap segment"):
        _resolve(factory, ids, {0: donor_id})
    with factory.begin() as session:
        session.get(Run, donor_run_id).segment_snapshot = session.get(Run, ids.run).segment_snapshot
        session.get(LibraryClip, library_id).local_path = ids.clips[1]
    with pytest.raises(script.ScriptCaptionsRefused, match="bytes differ"):
        _resolve(factory, ids, {0: donor_id})
    with factory.begin() as session:
        session.delete(session.get(LibraryClip, library_id))
    assert _resolve(factory, ids, {0: donor_id})["warnings"] == ["library_clip_missing"]
    with pytest.raises(script.ScriptCaptionsRefused, match="not a library entry"):
        _resolve(factory, ids, {1: donor_id})


def _baseline_lines(ids):
    probe = q.LocalMedia().probe
    return q.captions_from_script([
        {"source_start": 0.0, "source_end": 2.0, "clip_duration": probe(ids.clips[0]).duration_sec, "prompt": PROMPTS[0]},
        {"source_start": 2.0, "source_end": 3.5, "clip_duration": probe(ids.clips[1]).duration_sec, "prompt": PROMPTS[1]}])


def test_files_outside_the_data_root_are_refused(world, tmp_path):
    factory, ids = world
    outside = _clip(tmp_path / "outside.mp4", 2)
    with factory.begin() as session:
        session.get(RunSegment, ids.segments[0]).local_result_path = outside
    with pytest.raises(script.ScriptCaptionsRefused, match="outside the data root"):
        _resolve(factory, ids)


# ---------------------------------------------------------------------------
# Contract + Apply through the agent loop and the API
# ---------------------------------------------------------------------------

from tests.test_api_assistant import api, root  # noqa: E402,F401
from tests.test_assistant_turns import setup  # noqa: E402,F401
from tests.test_assistant_agent_loop import fixture_loop  # noqa: E402


def _evidence(setup, caption_id, *, lines_sha=None, lineage="run_segments", apply_allowed=True, old_lines=None):
    lines = [{"start": 0.0, "end": 1.2, "text": "Hola."}, {"start": 1.3, "end": 2.5, "text": "Adiós."}]
    final_sha = hashlib.sha256(setup.base.path.read_bytes()).hexdigest()
    seg = str(uuid.uuid4())
    return {"tool": "script_captions", "lineage": lineage, "final_sha256": final_sha, "final_duration_sec": 3.0,
            "pieces_duration_sec": 3.0,
            "pieces": [{"kind": "library" if lineage != "run_segments" else "segment", "segment_def_id": seg,
                        "run_segment_id": seg, "run_id": setup.base.run, "source_start_sec": 0.0,
                        "source_end_sec": 3.0, "clip_duration_sec": 3.0, "offset_sec": 0.0,
                        "clip_sha256": "a" * 64, "prompt_sha256": "b" * 64, "script_lines": 2}],
            "lines": lines, "lines_sha256": lines_sha or digest(lines),
            "captions": [{"caption_id": caption_id, "status": "done", "source_type": "run", "language": "es",
                          "lines_count": 1, "lines_sha256": digest(old_lines) if old_lines else None,
                          "apply_allowed": apply_allowed}],
            "captions_total": 1, "warnings": []}


OLD_LINES = [{"start": 0.0, "end": 1.0, "text": "transcribed"}]


def _caption(setup, **extra):
    with setup.factory.begin() as session:
        caption = RunCaption(run_id=setup.base.run, status=CaptionStatus.done, source_type="run", language="es",
                             lines=OLD_LINES, lines_source="transcribe", **extra)
        session.add(caption)
        session.flush()
        return caption.id


def _proposal_turn(setup, monkeypatch, *, lineage="run_segments"):
    from app.assistant_agent_contracts import FinalAction, ScriptCaptionsAction
    from app.assistant_planner import CaptionLinesReply
    caption_id = _caption(setup)
    svc, turn_id, broker, media, supervisor = fixture_loop(setup)
    evidence = _evidence(setup, caption_id, lineage=lineage, old_lines=OLD_LINES)
    media.script_captions = lambda **kwargs: evidence
    scope = {"run_id": setup.base.run, "base_version_id": setup.base.version}
    reply = CaptionLinesReply(kind="caption_lines", explanation="Script captions for the Spanish caption.",
                              caption_id=caption_id, lines_sha256=evidence["lines_sha256"], **scope)
    broker.actions[:] = [ScriptCaptionsAction(kind="script_captions", **scope),
                         FinalAction(kind="final", reply=reply, **scope)]
    supervisor.run_once()
    queued = []
    monkeypatch.setattr("app.tasks.enqueue_process_caption", lambda cid: queued.append(cid))
    monkeypatch.setattr("app.api_v2._spend_reserve", lambda *a, **k: None)
    monkeypatch.setattr("app.api_v2._admit_retained_run_source", lambda db, ids: None)
    return turn_id, caption_id, evidence, queued


def test_apply_is_one_transaction_and_once_per_turn(setup, api, monkeypatch):
    from app.assistant_turns import AssistantTurn
    client, _ = api
    turn_id, caption_id, evidence, queued = _proposal_turn(setup, monkeypatch)
    with setup.factory() as session:
        turn = session.get(AssistantTurn, turn_id)
        assert turn.status == "succeeded" and turn.reply["kind"] == "caption_lines"
    state = client.get(root(setup) + "/context").json()
    shown = next(t for t in state["turns"] if t["id"] == turn_id)
    assert shown["reply"]["caption_id"] == caption_id
    assert any((s["observation"] or {}).get("qa", {}).get("tool") == "script_captions" for s in shown["analysis"]["steps"])
    response = client.post(root(setup) + f"/turns/{turn_id}/apply-captions", json={})
    assert response.status_code == 200, response.text
    assert response.json()["repeated"] is False and queued == [caption_id]
    with setup.factory() as session:
        caption = session.get(RunCaption, caption_id)
        assert caption.lines == evidence["lines"] and caption.lines_source == "manual"
        assert getattr(caption.status, "value", caption.status) == "queued"
    again = client.post(root(setup) + f"/turns/{turn_id}/apply-captions", json={})
    assert again.status_code == 200 and again.json()["repeated"] is True and queued == [caption_id]


@pytest.mark.parametrize("mutate", ["lines", "status", "result"])
def test_apply_refuses_a_changed_caption_or_result(setup, api, monkeypatch, mutate):
    client, _ = api
    turn_id, caption_id, _, queued = _proposal_turn(setup, monkeypatch)
    with setup.factory.begin() as session:
        caption = session.get(RunCaption, caption_id)
        if mutate == "lines":
            caption.lines = [{"start": 0.0, "end": 1.0, "text": "edited by hand"}]
        elif mutate == "status":
            caption.status = CaptionStatus.burning
    if mutate == "result":
        setup.base.path.write_bytes(b"a different result")
    response = client.post(root(setup) + f"/turns/{turn_id}/apply-captions", json={})
    assert response.status_code == 409 and queued == []
    with setup.factory() as session:
        assert session.get(RunCaption, caption_id).lines_source == "transcribe"


def test_library_donor_needs_operator_attestation(setup, api, monkeypatch):
    client, _ = api
    turn_id, caption_id, _, queued = _proposal_turn(setup, monkeypatch, lineage="library_donor_attestation_required")
    refused = client.post(root(setup) + f"/turns/{turn_id}/apply-captions", json={})
    assert refused.status_code == 409 and "library clip" in refused.json()["detail"] and queued == []
    ok = client.post(root(setup) + f"/turns/{turn_id}/apply-captions", json={"attest_library_donor": True})
    assert ok.status_code == 200 and queued == [caption_id]


def test_caption_proposal_requires_this_turns_evidence():
    from app.assistant_agent_contracts import (AgentActionEnvelope, AgentObservation, AgentStepContext,
        validate_agent_action)
    from tests.test_assistant_planner import context, RUN, BASE, OTHER
    ns = SimpleNamespace(base=SimpleNamespace(path=SimpleNamespace(read_bytes=lambda: b"x"), run=RUN))
    evidence = _evidence(ns, OTHER, old_lines=OLD_LINES)
    final = AgentActionEnvelope.model_validate({"action": {"kind": "final", "run_id": RUN, "base_version_id": BASE,
        "reply": {"kind": "caption_lines", "run_id": RUN, "base_version_id": BASE, "explanation": "x",
                  "caption_id": OTHER, "lines_sha256": evidence["lines_sha256"]}}}).action
    with pytest.raises(ValueError, match="script_captions evidence"):
        validate_agent_action(final, AgentStepContext(context=context(), step_index=0))
    observed = AgentStepContext(context=context(), step_index=1,
        observations=(AgentObservation(kind="qa", action_index=0, qa=evidence),))
    assert validate_agent_action(final, observed).reply.caption_id == OTHER
    blocked = AgentStepContext(context=context(), step_index=1, observations=(AgentObservation(
        kind="qa", action_index=0, qa={**evidence, "captions": [{**evidence["captions"][0], "apply_allowed": False}]}),))
    with pytest.raises(ValueError, match="applicable caption"):
        validate_agent_action(final, blocked)
    with pytest.raises(ValueError):  # the digest must match the lines
        AgentObservation(kind="qa", action_index=0, qa={**evidence, "lines_sha256": "c" * 64})


def test_legacy_planner_schema_never_offers_caption_lines():
    import json
    from app.assistant_planner import _cli_reply_schema, _agent_schema
    assert "caption_lines" not in json.dumps(_cli_reply_schema())
    assert "caption_lines" in json.dumps(_agent_schema()) and "script_captions" in json.dumps(_agent_schema())


def test_manual_caption_writes_take_the_row_lock():
    """The lines/retry/delete endpoints re-read the caption under FOR UPDATE,
    so a write that raced the assistant's Apply sees its committed state."""
    import inspect
    from app import api_v2
    for name in ("update_caption_lines", "retry_caption", "delete_caption"):
        assert "_lock_caption_or_404(" in inspect.getsource(getattr(api_v2, name)), name
    statement = str(api_v2.select(RunCaption).where(RunCaption.id == "x").with_for_update()
                    .compile(dialect=__import__("sqlalchemy.dialects.postgresql", fromlist=["dialect"]).dialect()))
    assert "FOR UPDATE" in statement
