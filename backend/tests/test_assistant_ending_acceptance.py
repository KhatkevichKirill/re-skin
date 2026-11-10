"""Adversarial ending-extension acceptance; synthetic IDs/bytes only."""
from fractions import Fraction
import json

import pytest

from app.assistant_contracts import (
    EditPlan, ExtendEndingOperation, ending_bounds, validate_local_plan_capability,
)
from app.assistant_ending import resolve_ending_timing
from app.assistant_media import AssistantMediaError, MediaMetadata

_BASE = "11111111-1111-4111-8111-111111111111"
_RUN = "22222222-2222-4222-8222-222222222222"
_ASSET = "33333333-3333-4333-8333-333333333333"
_DONOR_VERSION = "44444444-4444-4444-8444-444444444444"


def ending(**changes):
    payload = dict(kind="extend_ending", start_sec=2., end_sec=4.,
        source={"asset_id": _ASSET, "version_id": _DONOR_VERSION},
        donor_start_sec=0., visual={"kind": "black"})
    payload.update(changes)
    return ExtendEndingOperation.model_validate(payload)


@pytest.mark.parametrize("duration,fps,changes", [
    (float(Fraction(1191*99841,999242)), "999242/99841",
        {"start_sec": float(Fraction(1181*99841,999242)), "end_sec": 120.}),
    (5., "999998/999999", {"start_sec": 4., "end_sec": 8.,
        "visual": {"kind": "freeze_selected", "timestamp_sec": 0.}, "fade_out_sec": 2.}),
    (2.99999999, "30", {"end_sec": 3.}),
    (2., "999998/999999", {"start_sec": 1., "end_sec": 11.}),
    (3., "30", {"visual": {"kind": "freeze_selected", "timestamp_sec": 1.99999999}}),
])
def test_tiny_frame_tolerance_cannot_authorize_extra_duration_or_freeze_frame(duration, fps, changes):
    operation = ending(**changes)
    with pytest.raises(ValueError):
        ending_bounds(operation, duration, fps)
    info = MediaMetadata(duration, duration, 160, 120, fps, "1:1", True)
    with pytest.raises(AssistantMediaError):
        resolve_ending_timing(info, start_sec=operation.start_sec, end_sec=operation.end_sec,
            donor_start_sec=operation.donor_start_sec, visual_kind=operation.visual.kind,
            freeze_timestamp_sec=getattr(operation.visual, "timestamp_sec", None),
            fade_out_sec=operation.fade_out_sec, donor_audio_duration_sec=120.)


@pytest.mark.parametrize("kind", ["trim", "cut", "image_fade", "replace_footage", "audio_splice"])
def test_extension_exception_does_not_relax_old_operation_duration_bounds(kind):
    operation = {"kind": kind, "start_sec": 2., "end_sec": 4.}
    if kind in ("replace_footage", "audio_splice"):
        operation["source"] = {"asset_id": _ASSET, "version_id": _DONOR_VERSION}
    plan = EditPlan.model_validate({"run_id": _RUN, "base_version_id": _BASE, "operations": [operation]})
    with pytest.raises(ValueError):
        validate_local_plan_capability(plan, duration_sec=3., fps="30")


from datetime import datetime, timezone
import uuid
from types import SimpleNamespace

from sqlalchemy import func, select

from app import assistant_audio, assistant_executor
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_agent_contracts import (
    CandidateEvidence, FinalAction, InspectFramesAction, RenderCandidateAction,
    TranscribeWindowAction, plan_fingerprint,
)
from app.assistant_audio import AudioMetadata
from app.assistant_jobs import AssistantJob, AssistantJobOutbox, accept_job, claim_job, fail_job
from app.assistant_models import AssistantArtifact, AssistantMessage, AssistantSelection, AssistantVersion
from app.assistant_planner import PlannerError, ProposeReply, _reply
from app.assistant_service import AssistantAdmissionService, AssistantCapabilityError
from app.assistant_supervisor import AssistantSupervisor
from app.assistant_turns import AssistantTurn, TurnConflict, claim_turn
from app.models import Run
from tests.test_assistant_agent_loop import Broker, DIGEST, JPEG
from tests.test_api_assistant import api, root
from tests.test_assistant_turns import NOW, service, setup


def scoped_plan(context, **changes):
    operation = ending(**changes).model_dump(mode="json")
    operation["source"] = {"asset_id": context.donor.artifact, "version_id": context.donor.version}
    return EditPlan(run_id=context.base.run, base_version_id=context.base.version, operations=[operation])


def admission(context, policy=None):
    return AssistantAdmissionService(storage=context.storage,
        policy=policy or ProjectAllowlist([context.base.project, context.donor.project]), enabled=True)


def counts(session):
    return tuple(session.scalar(select(func.count()).select_from(model))
        for model in (AssistantJob, AssistantJobOutbox, AssistantVersion, AssistantMessage))


def submit(context, session, plan, *, proof=None, key="accepted-ending"):
    return admission(context).admit(session, run_id=context.base.run, plan=plan,
        instruction="Replace and extend the ending", idempotency_key=key, actor="operator", now=NOW,
        explicit_base_version_id=context.base.version, verified_candidate_turn_id=proof)


@pytest.mark.parametrize("existing_job", [False, True])
def test_direct_or_idempotent_admission_cannot_skip_durable_candidate_proof(setup, monkeypatch, existing_job):
    plan = scoped_plan(setup)
    svc = admission(setup)
    monkeypatch.setattr(svc, "_validate", lambda *args, **kwargs: pytest.fail("unverified ending reached media probes"))
    with setup.factory.begin() as session:
        if existing_job:
            accept_job(session, plan, request_scope=svc._scope(setup.base.run),
                idempotency_key="unverified", actor="operator", now=NOW)
        before = counts(session)
        with pytest.raises(AssistantCapabilityError, match="verified agent candidate"):
            svc.admit(session, run_id=setup.base.run, plan=plan, instruction="Extend", idempotency_key="unverified",
                actor="operator", now=NOW, explicit_base_version_id=setup.base.version)
        assert counts(session) == before


@pytest.mark.parametrize("visual_kind", ["black", "freeze_selected"])
def test_real_one_frame_extension_accepts_canonical_added_frame_but_requires_it(tmp_path, visual_kind):
    """Actual 91->92-frame render/inspection; speech windows are synthetic."""
    import base64
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from app import assistant_media as media
    from app.assistant_agent_contracts import (
        AgentObservation, AgentStepContext, CandidateSpeechEvidence, FrameEvidence,
        candidate_visual_windows, validate_agent_action,
    )
    from app.assistant_agent_media import AgentMediaService
    from app.assistant_planner import PlannerSpeechEvidence
    from app.assistant_store import AssistantArtifactStorage, ArtifactInput, _add_artifact, get_version_artifact, import_run_version
    from app.db import Base
    from app.models import VideoProject
    from tests.test_assistant_ending import donor, video

    data = tmp_path / "owned-data"
    data.mkdir()
    source = video(data / "base.mp4", fps="30000/1001", frames=91)
    audio = donor(data / "donor.wav")
    storage = AssistantArtifactStorage(data)
    engine = create_engine(f"sqlite:///{tmp_path / 'one-frame.db'}")
    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    rate = Fraction(30000, 1001)
    start, base_end, end = (float(Fraction(frame, 1) / rate) for frame in (90, 91, 92))
    try:
        with factory.begin() as session:
            project = VideoProject(source_type="upload", source_ref="owned fixture", status="ready")
            session.add(project)
            session.flush()
            run = Run(project_id=project.id, status="done", result_local_path=str(source),
                result_duration_sec=media.probe_trusted_media(source).video_duration_sec)
            session.add(run)
            session.flush()
            version = import_run_version(session, run, actor="operator", storage=storage)
            asset = get_version_artifact(session, version.id)
            audio_asset = _add_artifact(session, version,
                ArtifactInput(audio, "audio", {"duration_sec": 6., "has_audio": True}), storage)
            run_id, version_id, asset_id, donor_id, project_id = run.id, version.id, asset.id, audio_asset.id, project.id
        policy = ProjectAllowlist([project_id])
        from app.assistant_turns import AssistantTurnService
        svc = AssistantTurnService(storage=storage, policy=policy, enabled=True, agent_enabled=True)
        with factory() as session:
            context = svc.build_context(session, run_id=run_id, base_version_id=version_id,
                instruction="One actual new frame", source_refs=((donor_id, version_id),))
        assert context.selected.duration_sec == 3.036367
        assert base_end < context.selected.duration_sec
        visual = {"kind": visual_kind}
        options = {}
        if visual_kind == "freeze_selected":
            visual["timestamp_sec"] = 0.
            # A tolerated raw duration just below two frames must still include
            # the canonical first fade frame, which is also the replacement seam.
            options["fade_out_sec"] = float(2/rate) - 1e-10
        plan = EditPlan(run_id=run_id, base_version_id=version_id, operations=[{
            "kind": "extend_ending", "start_sec": start, "end_sec": end,
            "source": {"asset_id": donor_id, "version_id": version_id},
            "donor_start_sec": 0., "visual": visual, **options}])

        with AgentMediaService(factory, storage=storage, policy=policy, target_version_id=version_id) as tools:
            source_times = [float(Fraction(89, 1)/rate), start]
            if visual_kind == "freeze_selected":
                source_times.insert(0, 0.)
            original_frames = tools.inspect_frames(asset_id=asset_id, version_id=version_id,
                timestamps_sec=source_times)["frames"]
            candidate = CandidateEvidence.model_validate(tools.render_candidate(plan)["candidate"])
            inspected = tools.inspect_frames(candidate_id=candidate.candidate_id,
                timestamps_sec=[start, base_end])["frames"]
            assert inspected[-1]["timestamp_sec"] == base_end
            assert inspected[-1]["timestamp_sec"] < context.selected.duration_sec

            def frames(records):
                return tuple(FrameEvidence.model_validate({
                    **{key: value for key, value in item.items() if key in FrameEvidence.model_fields},
                    "data_base64": base64.b64encode(tools.read_frame(item["frame_id"])).decode(),
                }) for item in records)

            donor_end = float(Fraction(round(Fraction(92, 1)/rate*48000)
                - round(Fraction(90, 1)/rate*48000), 48000))
            speech = PlannerSpeechEvidence(asset_id=donor_id, version_id=version_id,
                source_start_sec=0., source_end_sec=donor_end, source_duration_sec=6.,
                language="en", language_probability=.99, words=(), coverage_complete=False)
            candidate_path = tools.candidates[candidate.candidate_id]["path"]
            audio_end = assistant_audio.probe_trusted_audio(candidate_path).duration_sec
            candidate_speech = CandidateSpeechEvidence(candidate_id=candidate.candidate_id,
                source_sha256=candidate.sha256, source_start_sec=start, source_end_sec=audio_end,
                source_duration_sec=audio_end, language="en", language_probability=.99,
                words=(), coverage_complete=False)
            observations = (
                AgentObservation(kind="frames", action_index=0, frames=frames(original_frames)),
                AgentObservation(kind="speech", action_index=1, speech=(speech,)),
                AgentObservation(kind="candidate", action_index=2, candidate=candidate),
                AgentObservation(kind="frames", action_index=3, frames=frames(inspected)),
                AgentObservation(kind="speech", action_index=4, candidate_speech=(candidate_speech,)),
            )
            step = AgentStepContext(context=context, step_index=5, observations=observations)
            action = FinalAction(kind="final", run_id=run_id, base_version_id=version_id,
                candidate_id=candidate.candidate_id, reply=ProposeReply(kind="propose", run_id=run_id,
                    base_version_id=version_id, explanation="One bounded new frame", plan=plan))
            added_window = candidate_visual_windows(plan, context, candidate.duration_sec)[2]
            assert added_window == (base_end, end)
            assert validate_agent_action(action, step) == action

            # All source/speech/candidate bindings remain valid, but the actual
            # added frame is absent. No epsilon may let the old seam frame pass.
            old_only = observations[3].model_copy(update={"frames": observations[3].frames[:1]})
            missing = AgentStepContext(context=context, step_index=5,
                observations=(*observations[:3], old_only, observations[4]))
            with pytest.raises(ValueError, match="candidate frames"):
                validate_agent_action(action, missing)
    finally:
        engine.dispose()



def test_donor_evidence_uses_exact_frame_sample_clock_at_half_sample_ties(setup):
    import base64
    from app.assistant_agent_contracts import AgentObservation, AgentStepContext, FrameEvidence, ending_source_evidence
    from app.assistant_planner import PlannerContext, PlannerSpeechEvidence

    # At 96000/1001 FPS, frame3 is sample1501.5 and rounds to1502;
    # float(start_sec)*48000 can land just below the tie and round to1501.
    rate = Fraction(96000, 1001)
    start, base_end, end = (float(Fraction(frame, 1) / rate) for frame in (3, 300, 400))
    plan = scoped_plan(setup, start_sec=start, end_sec=end)
    svc = service(setup)
    with setup.factory() as session:
        original = svc.build_context(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Rational timing", source_refs=((setup.donor.artifact, setup.donor.version),))
        source_hash = session.get(AssistantArtifact, setup.base.artifact).sha256
    context = PlannerContext.model_validate({**original.model_dump(mode="python"),
        "selected": original.selected.model_copy(update={"duration_sec": base_end, "fps": str(rate)}),
        "sources": (original.sources[0].model_copy(update={"duration_sec": 5.}),)})
    frames = tuple(FrameEvidence(frame_id=str(uuid.uuid4()), asset_id=setup.base.artifact,
        version_id=setup.base.version, timestamp_sec=timestamp, width=4, height=4,
        source_sha256=source_hash, sha256=DIGEST, size_bytes=len(JPEG),
        data_base64=base64.b64encode(JPEG).decode()) for timestamp in (0., start))
    expected_donor_end = float(Fraction(200200-1502, 48000))
    speech = PlannerSpeechEvidence(asset_id=setup.donor.artifact, version_id=setup.donor.version,
        source_start_sec=0., source_end_sec=expected_donor_end, source_duration_sec=5.,
        language="en", language_probability=.99, words=(), coverage_complete=False)
    step = AgentStepContext(context=context, step_index=2, observations=(
        AgentObservation(kind="frames", action_index=0, frames=frames),
        AgentObservation(kind="speech", action_index=1, speech=(speech,))))
    assert ending_source_evidence(plan, step)



def test_legacy_reply_parser_cannot_offer_ending_without_agent_capability(setup):
    svc = service(setup)
    with setup.factory() as session:
        context = svc.build_context(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Extend", source_refs=((setup.donor.artifact, setup.donor.version),))
    proposal = ProposeReply(kind="propose", run_id=setup.base.run, base_version_id=setup.base.version,
        explanation="Existing donor ending", plan=scoped_plan(setup))
    raw = json.dumps({"type": "result", "subtype": "success", "is_error": False,
        "structured_output": {"reply": proposal.model_dump(mode="json")}}).encode()
    with pytest.raises(PlannerError):
        _reply(raw, context)


def test_api_accept_and_retry_keep_atomic_ending_proof_after_linking_edit_job(setup, api, monkeypatch):
    trial = verified_workflow(setup, monkeypatch)
    client, _ = api
    accept_url = root(setup) + f"/turns/{trial.turn_id}/accept"
    first = client.post(accept_url, json={"operation_indexes": [0]})
    assert first.status_code == 202, first.text
    original_job = first.json()["job"]["id"]
    assert client.post(accept_url).json()["job"]["id"] == original_job
    with setup.factory.begin() as session:
        turn = session.get(AssistantTurn, trial.turn_id)
        assert turn.edit_job_id == original_job
        assert admission(setup).revalidate_job(session, original_job).job_id == original_job
        now = datetime.now(timezone.utc)
        lease = claim_job(session, original_job, owner="acceptance", now=now)
        fail_job(session, lease, error_code="controlled_local_failure", now=now)
    retry_url = root(setup) + f"/turns/{trial.turn_id}/retry-edit"
    retried = client.post(retry_url, json={"idempotency_key": "retry-ending"})
    assert retried.status_code == 202, retried.text
    new_job = retried.json()["job"]["id"]
    assert new_job != original_job
    assert client.post(retry_url, json={"idempotency_key": "retry-ending"}).json()["job"]["id"] == new_job
    assert "storage_key" not in retried.text and str(setup.data) not in retried.text
    with setup.factory() as session:
        turn = session.get(AssistantTurn, trial.turn_id)
        assert turn.edit_job_id == new_job
        receipt = admission(setup).revalidate_job(session, new_job)
        assert receipt.base_version_id == setup.base.version
        assert session.get(AssistantJob, new_job).accepted_plan == trial.plan.model_dump(mode="json")
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 2



def test_low_level_queued_extension_is_failed_before_renderer_and_never_published(setup, monkeypatch):
    plan = scoped_plan(setup)
    with setup.factory.begin() as session:
        job = accept_job(session, plan, request_scope=admission(setup)._scope(setup.base.run),
            idempotency_key="raw-queue", actor="operator", now=NOW)
        job_id, version_id = job.id, job.pending_version_id
        selection = session.get(AssistantSelection, setup.base.conversation)
        before = (selection.selected_version_id, selection.latest_version_id, selection.epoch)
    monkeypatch.setattr(assistant_executor, "render_local_candidate",
        lambda *args, **kwargs: pytest.fail("missing proof reached renderer"))
    result = assistant_executor.execute_assistant_job(setup.factory, job_id, owner="acceptance",
        storage=setup.storage, policy=ProjectAllowlist([setup.base.project, setup.donor.project]), enabled=True)
    assert result.status == "failed"
    with setup.factory() as session:
        assert session.get(AssistantJob, job_id).status == "failed"
        assert session.get(AssistantVersion, version_id).status == "failed"
        selection = session.get(AssistantSelection, setup.base.conversation)
        assert (selection.selected_version_id, selection.latest_version_id, selection.epoch) == before
        assert session.scalar(select(func.count()).select_from(AssistantArtifact)) == 3
        assert session.get(Run, setup.base.run).result_local_path == str(setup.base.path)
    assert setup.base.path.read_bytes() == b"fixture-0"


@pytest.mark.parametrize("agent_enabled", [False, True])
def test_text_only_finish_cannot_publish_an_extension_proposal(setup, agent_enabled):
    svc = service(setup, agent_enabled=agent_enabled)
    with setup.factory.begin() as session:
        turn = svc.accept(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Extend", idempotency_key="text-only", actor="operator", now=NOW,
            source_refs=((setup.donor.artifact, setup.donor.version),))
        lease = claim_turn(session, turn.id, owner="text-only", now=NOW)
        before = counts(session)
        with pytest.raises(TurnConflict):
            svc.finish(session, lease, ProposeReply(kind="propose", run_id=setup.base.run,
                base_version_id=setup.base.version, explanation="Unverified extension", plan=scoped_plan(setup)), now=NOW)
        assert turn.reply is None and turn.status == "running"
        assert counts(session) == before


def verified_workflow(context, monkeypatch, *, missing=None):
    """Use the actual durable loop and validators; only provider/media are fakes."""
    svc = service(context, agent_enabled=True)
    plan = scoped_plan(context)
    with context.factory.begin() as session:
        turn = svc.accept(session, run_id=context.base.run, base_version_id=context.base.version,
            instruction="Extend from two to four seconds", idempotency_key="ending-workflow", actor="operator",
            now=NOW, source_refs=((context.donor.artifact, context.donor.version),))
        turn_id = turn.id
        source_hash = session.get(AssistantArtifact, context.base.artifact).sha256
    path = context.data / "owned-retained-ending.bin"
    path.write_bytes(b"owned retained ending candidate")
    pinned = context.storage.pin(path)
    candidate = CandidateEvidence(candidate_id=str(uuid.uuid4()), base_version_id=context.base.version,
        plan=plan, plan_sha256=plan_fingerprint(plan), sha256=pinned.sha256, size_bytes=pinned.size_bytes,
        duration_sec=4., has_audio=True, subtitle_status="absent", mixed_audio_changed=True)
    scope = dict(run_id=context.base.run, base_version_id=context.base.version)

    class Media:
        renders = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def inspect_frames(self, **kwargs):
            times = kwargs["timestamps_sec"]
            if missing == "extended_frames" and kwargs.get("candidate_id"):
                times = (2., 2.9)
            return {"frames": [dict(frame_id=str(uuid.uuid4()), candidate_id=kwargs.get("candidate_id"),
                asset_id=kwargs.get("asset_id"), version_id=kwargs.get("version_id"),
                timestamp_sec=value, width=4, height=4,
                source_sha256=candidate.sha256 if kwargs.get("candidate_id") else source_hash,
                sha256=DIGEST, size_bytes=len(JPEG), mime_type="image/jpeg") for value in times]}
        def read_frame(self, frame_id): return JPEG
        def transcribe_window(self, **kwargs):
            is_candidate = kwargs.get("candidate_id") is not None
            common = dict(source_start_sec=kwargs["start_sec"],
                source_end_sec=3. if is_candidate and missing == "extended_speech" else kwargs["end_sec"],
                source_duration_sec=4. if is_candidate else 3., language="en", language_probability=.99,
                words=(), words_truncated=False, coverage_complete=False, timing_precision="estimated_word",
                pronunciation_verified=False, speaker_verified=False, review_required=True)
            if is_candidate:
                return {"candidate_speech": [dict(candidate_id=candidate.candidate_id,
                    source_sha256=candidate.sha256, **common)]}
            return {"speech": [dict(asset_id=context.donor.artifact, version_id=context.donor.version, **common)]}
        def render_candidate(self, selected):
            self.renders += 1
            assert selected == plan
            return {"candidate": candidate.model_dump(mode="json")}
        def persist_candidate(self, candidate_id):
            assert candidate_id == candidate.candidate_id
            return dict(storage_key=pinned.storage_key, sha256=pinned.sha256, size_bytes=pinned.size_bytes)

    broker = Broker([
        InspectFramesAction(kind="inspect_frames", **scope, asset_id=context.base.artifact,
            version_id=context.base.version, timestamps_sec=(1.9333333333333333, 2.)),
        TranscribeWindowAction(kind="transcribe_window", **scope, asset_id=context.donor.artifact,
            version_id=context.donor.version, start_sec=0., end_sec=2.),
        RenderCandidateAction(kind="render_candidate", **scope, plan=plan),
        InspectFramesAction(kind="inspect_frames", **scope, candidate_id=candidate.candidate_id,
            timestamps_sec=(2., 3.2, 3.966666666666667)),
        TranscribeWindowAction(kind="transcribe_window", **scope, candidate_id=candidate.candidate_id,
            start_sec=2., end_sec=4.),
        FinalAction(kind="final", **scope, candidate_id=candidate.candidate_id,
            reply=ProposeReply(kind="propose", **scope, explanation="Bounded verified ending", plan=plan)),
    ])
    media = Media()
    supervisor = AssistantSupervisor(context.factory, service=svc, broker=broker, owner="ending-acceptance",
        execute=lambda *args, **kwargs: None, clock=lambda: NOW, sleep=lambda _: None,
        media_factory=lambda **kwargs: media)
    supervisor.run_once()
    monkeypatch.setattr(assistant_audio, "probe_trusted_audio", lambda *args, **kwargs: AudioMetadata(3., 48000, 1, "mono"))
    return SimpleNamespace(service=svc, plan=plan, turn_id=turn_id, pinned=pinned, candidate=candidate, media=media)


def test_real_durable_workflow_admits_idempotently_and_revalidates_exact_binding(setup, monkeypatch):
    trial = verified_workflow(setup, monkeypatch)
    with setup.factory.begin() as session:
        assert session.get(AssistantTurn, trial.turn_id).status == "succeeded"
        receipt = submit(setup, session, trial.plan, proof=trial.turn_id)
        assert submit(setup, session, trial.plan, proof=trial.turn_id) == receipt
        assert admission(setup).revalidate_job(session, receipt.job_id) == receipt
        message = session.get(AssistantMessage, receipt.message_id)
        assert message.context_snapshot["verified_candidate_turn_id"] == trial.turn_id
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 1
        assert session.get(AssistantVersion, receipt.pending_version_id).status == "pending"


@pytest.mark.parametrize("changes", [
    {"donor_start_sec": .5}, {"end_sec": 4.5},
    {"visual": {"kind": "freeze_selected", "timestamp_sec": 0.}},
])
def test_verified_turn_does_not_authorize_a_different_atomic_plan(setup, monkeypatch, changes):
    trial = verified_workflow(setup, monkeypatch)
    with setup.factory.begin() as session:
        before = counts(session)
        with pytest.raises(TurnConflict):
            submit(setup, session, scoped_plan(setup, **changes), proof=trial.turn_id)
        assert counts(session) == before


@pytest.mark.parametrize("revocation", ["policy", "run_owner"])
def test_accepted_extension_revalidates_live_donor_authority_before_execution(setup, monkeypatch, revocation):
    trial = verified_workflow(setup, monkeypatch)
    with setup.factory.begin() as session:
        receipt = submit(setup, session, trial.plan, proof=trial.turn_id)
        policy = ProjectAllowlist([setup.base.project, setup.donor.project])
        if revocation == "policy":
            policy = ProjectAllowlist([setup.base.project])
        else:
            session.get(Run, setup.donor.run).project_id = setup.base.project
            session.flush()
        before = counts(session)
        with pytest.raises(AssistantAccessDenied):
            admission(setup, policy=policy).revalidate_job(session, receipt.job_id)
        assert counts(session) == before
        assert session.get(AssistantJob, receipt.job_id).status == "queued"


@pytest.mark.parametrize("missing", ["extended_frames", "extended_speech"])
def test_missing_extended_evidence_fails_durable_turn_and_cannot_be_admitted(setup, monkeypatch, missing):
    trial = verified_workflow(setup, monkeypatch, missing=missing)
    with setup.factory.begin() as session:
        assert session.get(AssistantTurn, trial.turn_id).status == "failed"
        with pytest.raises(AssistantCapabilityError, match="completed agent proposal"):
            submit(setup, session, trial.plan, proof=trial.turn_id)
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 0


@pytest.mark.parametrize("changed", ["candidate", "donor"])
def test_changed_retained_or_donor_bytes_invalidate_candidate_authority(setup, monkeypatch, changed):
    trial = verified_workflow(setup, monkeypatch)
    with setup.factory.begin() as session:
        if changed == "candidate":
            path = setup.storage.path_for(trial.pinned.storage_key)
        else:
            artifact = session.get(AssistantArtifact, setup.donor.artifact)
            path = setup.storage.path_for(artifact.storage_key)
        path.chmod(0o600)
        path.write_bytes(b"X" * path.stat().st_size)
        before = counts(session)
        with pytest.raises(ValueError):
            submit(setup, session, trial.plan, proof=trial.turn_id)
        assert counts(session) == before


