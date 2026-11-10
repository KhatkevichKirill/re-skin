"""Offline speech context integration using a fake credential-free sidecar."""
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DatabaseError
from tests.test_assistant_turns import setup, service, accept, NOW
from app import assistant_audio, assistant_media, assistant_legacy
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_asr import ASRTranscript, ASRWord
from app.assistant_broker_client import BrokerResult, context_fingerprint
from app.assistant_contracts import EditPlan, TimeRange
from app.assistant_enrichment import AssistantContextEnricher
from app.assistant_jobs import LostLease
from app.assistant_models import AssistantConversation
from app.assistant_planner import PlannerContext, PlannerSpeechEvidence, PlannerSpeechWord, ProposeReply
from app.assistant_supervisor import AssistantSupervisor
from app.assistant_turns import AssistantTurn, AssistantTurnService, claim_turn, fail_turn, request_cancel
from app.models import Run


@pytest.fixture(autouse=True)
def fake_audio_probe(monkeypatch):
    monkeypatch.setattr(assistant_audio, "probe_trusted_audio", lambda path, **kwargs:
        assistant_audio.AudioMetadata(assistant_media.probe_trusted_media(path).duration_sec, 48000, 2, "stereo"))


class ASR:
    def __init__(self, *, duration=3, fail=False, on_call=None):
        self.calls, self.duration, self.fail, self.on_call = [], duration, fail, on_call
    def transcribe(self, path, *, start_sec, end_sec, language, timeout_sec, cancelled):
        self.calls.append((Path(path), start_sec, end_sec, timeout_sec))
        cancelled()
        if self.on_call:
            self.on_call()
        if self.fail:
            raise OSError("private sidecar socket path failure")
        return ASRTranscript(language="en", language_probability=.95, network_denied=True, max_rss_kib=700000,
            source_start_sec=start_sec, source_end_sec=end_sec, source_duration_sec=self.duration,
            words=(ASRWord(text="market", start_sec=start_sec+.05, end_sec=start_sec+.3, probability=.91),
                   ASRWord(text="performance", start_sec=start_sec+.35, end_sec=min(end_sec, start_sec+.8), probability=.87)))


def context(setup, session, **kwargs):
    turn = accept(setup, session, instruction="Fix pronunciation of market performance", **kwargs)
    return PlannerContext.model_validate(turn.context_snapshot)


def enrich(setup, value, asr, **kwargs):
    return AssistantContextEnricher(setup.factory, service=service(setup), asr=asr, **kwargs).enrich(value)


def test_actual_sidecar_adapter_enriches_target_and_auto_discovered_donor_with_review_flags(setup):
    with setup.factory.begin() as session:
        original = context(setup, session, selected_passage=TimeRange(start_sec=.5, end_sec=2))
    asr = ASR()
    enriched = enrich(setup, original, asr)
    assert len(asr.calls) == 2
    assert asr.calls[0][1:3] == (.5, 2)
    assert {item.asset_id for item in enriched.speech_evidence} == {setup.base.artifact, setup.donor.artifact}
    assert {item.asset_id for item in enriched.sources} == {setup.donor.artifact}
    assert enriched.speech_evidence[0].words[0].start_sec == .55
    assert enriched.speech_evidence[0].coverage_complete is False
    assert all(not item.pronunciation_verified and not item.speaker_verified and item.review_required for item in enriched.speech_evidence)
    assert all(path.is_relative_to(setup.storage.root) for path, *_ in asr.calls)
    assert str(setup.data) not in enriched.model_dump_json() and "storage_key" not in enriched.model_dump_json()
    assert len(enriched.model_dump_json().encode()) < 65536


def test_nonlocalization_and_cached_evidence_never_repeat_asr(setup):
    with setup.factory.begin() as session:
        plain_turn = accept(setup, session, instruction="Trim ending")
        value = context(setup, session, idempotency_key="audio")
    asr = ASR()
    plain = PlannerContext.model_validate(plain_turn.context_snapshot)
    assert enrich(setup, plain, asr) == plain and asr.calls == []
    enriched = enrich(setup, value, asr)
    count = len(asr.calls)
    assert enrich(setup, enriched, asr) == enriched and len(asr.calls) == count


def test_missing_sidecar_produces_explicit_limitations_without_private_error(setup):
    with setup.factory.begin() as session:
        value = context(setup, session)
    enriched = enrich(setup, value, ASR(fail=True))
    assert enriched.speech_evidence == ()
    assert any("unavailable" in item for item in enriched.evidence_limitations)
    assert "private sidecar" not in enriched.model_dump_json()


def test_long_target_has_explicit_coverage_and_deadline_discards_late_result(setup, monkeypatch):
    monkeypatch.setattr(assistant_media, "probe_trusted_media", lambda *args, **kwargs:
        assistant_media.MediaMetadata(120, 120, 160, 120, "30", "1:1", True))
    with setup.factory.begin() as session:
        value = context(setup, session)
    asr = ASR(duration=120)
    enriched = enrich(setup, value, asr)
    assert all(call[2]-call[1] <= 60 for call in asr.calls)
    assert all(not item.coverage_complete for item in enriched.speech_evidence)
    assert any("other speech was not analyzed" in item for item in enriched.evidence_limitations)
    ticks = [0]
    late = ASR(duration=120, on_call=lambda: ticks.__setitem__(0, 91))
    bounded = enrich(setup, value, late, monotonic=lambda: ticks[0])
    assert len(late.calls) == 1 and bounded.speech_evidence == ()
    assert any("budget" in item for item in bounded.evidence_limitations)


def test_scope_revocation_blocks_target_before_asr_and_does_not_leak_donor(setup):
    with setup.factory.begin() as session:
        value = context(setup, session)
    denied = AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist(), enabled=True)
    asr = ASR()
    with pytest.raises(AssistantAccessDenied):
        AssistantContextEnricher(setup.factory, service=denied, asr=asr).enrich(value)
    assert asr.calls == []


def test_automatic_import_is_scoped_bounded_and_keeps_legacy_rows_unchanged(setup, monkeypatch):
    monkeypatch.setattr(assistant_legacy, "probe_trusted_media", assistant_media.probe_trusted_media)
    with setup.factory.begin() as session:
        value = context(setup, session)
        created = []
        for project in (setup.base.project, setup.donor.project, setup.denied.project):
            run = Run(project_id=project, status="done", result_local_path=str(setup.base.path), result_duration_sec=None)
            session.add(run)
            session.flush()
            created.append(run.id)
    result = enrich(setup, value, ASR())
    with setup.factory() as session:
        imported = set(session.scalars(select(AssistantConversation.run_id).where(AssistantConversation.run_id.in_(created))).all())
        assert imported == set(created[:2])
        assert all(session.get(Run, run_id).result_duration_sec is None for run_id in created)
    assert any("at most two" in item for item in result.evidence_limitations)


def test_enrichment_is_write_once_lease_fenced_and_retry_reuses_exact_context(setup):
    with setup.factory.begin() as session:
        original = context(setup, session)
        turn_id = session.scalar(select(AssistantTurn.id))
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
    enriched = enrich(setup, original, ASR())
    with setup.factory.begin() as session:
        saved = service(setup).save_enrichment(session, lease, enriched, now=NOW)
        turn = session.get(AssistantTurn, turn_id)
        assert turn.context_snapshot == original.model_dump(mode="json")
        assert turn.enrichment_snapshot["sha256"] == context_fingerprint(saved)
        assert service(setup).load_context(session, lease, now=NOW) == enriched
        with pytest.raises(DatabaseError), session.begin_nested():
            session.execute(text("UPDATE assistant_turns SET enrichment_snapshot = '{}' WHERE id=:id"), {"id": turn_id})
        changed = enriched.model_copy(update={"evidence_limitations": ("Changed",)})
        with pytest.raises(ValueError):
            service(setup).save_enrichment(session, lease, changed, now=NOW)
        fail_turn(session, lease, error_code="fixture", now=NOW)
        retried = service(setup).retry(session, run_id=setup.base.run, turn_id=turn_id, idempotency_key="retry", now=NOW)
        assert PlannerContext.model_validate(retried.context_snapshot) == enriched


def test_cancelled_turn_cannot_persist_optional_enrichment(setup):
    with setup.factory.begin() as session:
        original = context(setup, session)
        turn_id = session.scalar(select(AssistantTurn.id))
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        request_cancel(session, turn_id, now=NOW)
        with pytest.raises(LostLease):
            service(setup).save_enrichment(session, lease, original, now=NOW)
        assert session.get(AssistantTurn, turn_id).enrichment_snapshot is None


def test_enriched_donor_is_accepted_by_fenced_supervisor_and_native_context_is_durable_before_submit(setup):
    with setup.factory.begin() as session:
        original = context(setup, session)
        turn_id = session.scalar(select(AssistantTurn.id))
    class Broker:
        def submit(self, identity, value):
            self.fingerprint = context_fingerprint(value)
            with setup.factory() as session:
                row = session.get(AssistantTurn, identity)
                assert row.enrichment_snapshot["sha256"] == context_fingerprint(value)
            donor = value.sources[0]
            plan = EditPlan.model_validate({"run_id": value.run_id, "base_version_id": value.base_version_id,
                "operations": [{"kind": "audio_splice", "start_sec": 1, "end_sec": 2,
                    "source": {"asset_id": donor.asset_id, "version_id": donor.version_id}}]})
            return BrokerResult(turn_id=identity, context_sha256=context_fingerprint(value), state="succeeded",
                reply=ProposeReply(kind="propose", run_id=value.run_id, base_version_id=value.base_version_id,
                    explanation="Candidate mixed soundtrack window; listening review required.", plan=plan))
        def acknowledge(self, identity):
            return BrokerResult(turn_id=identity, context_sha256=self.fingerprint, state="succeeded", acknowledged=True)
    enricher = AssistantContextEnricher(setup.factory, service=service(setup), asr=ASR())
    worker = AssistantSupervisor(setup.factory, service=service(setup), broker=Broker(), owner="worker", enricher=enricher,
        clock=lambda: NOW, execute=lambda *args, **kwargs: None)
    worker.run_once()
    with setup.factory() as session:
        turn = session.get(AssistantTurn, turn_id)
        assert turn.status == "succeeded" and turn.reply["plan"]["operations"][0]["source"]["asset_id"] == setup.donor.artifact


def test_known_original_library_outro_is_frozen_and_not_inferred_from_current_run(setup):
    from app.assistant_store import import_run_version
    import uuid
    with setup.factory.begin() as session:
        run = Run(project_id=setup.base.project, status="done", result_local_path=str(setup.base.path),
            result_duration_sec=3, stitch_plan=[{"kind": "library", "clip_id": str(uuid.uuid4())}])
        session.add(run)
        session.flush()
        version = import_run_version(session, run, actor="operator", storage=setup.storage)
        run.stitch_plan = None
        session.flush()
        data = service(setup).version_context(session, run_id=run.id, base_version_id=version.id)
        assert data["settings"].original_has_library_outro is True
        unknown = service(setup).version_context(session, run_id=setup.base.run, base_version_id=setup.base.version)
        assert unknown["settings"].original_has_library_outro is None


def test_cached_donor_transcript_and_derived_messages_are_redacted_after_scope_revocation(setup):
    from app.api_assistant import get_context, get_turn, cancel_turn
    with setup.factory.begin() as session:
        original = context(setup, session)
        turn_id = session.scalar(select(AssistantTurn.id))
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
    enriched = enrich(setup, original, ASR())
    with setup.factory.begin() as session:
        service(setup).save_enrichment(session, lease, enriched, now=NOW)
        response = ProposeReply(kind="propose", run_id=original.run_id, base_version_id=original.base_version_id,
            explanation="Private donor speech: example-secret", plan=EditPlan.model_validate({"run_id": original.run_id,
                "base_version_id": original.base_version_id, "operations": [{"kind": "trim", "start_sec": 0, "end_sec": 2}]}))
        service(setup).finish(session, lease, response, now=NOW)
    restricted = AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist([setup.base.project]), enabled=True)
    with setup.factory.begin() as session:
        turn = get_turn(setup.base.run, turn_id, db=session, service=restricted)
        assert turn["source_access_changed"] and turn["speech_evidence"] == [] and turn["reply"] is None
        history = get_context(setup.base.run, db=session, service=restricted)
        serialized = json.dumps(history, default=str)
        assert setup.donor.artifact not in serialized and "example-secret" not in serialized
        assert [message["role"] for message in history["messages"]] == ["operator"]
        # Own control remains available and uses a redacted projection.
        assert cancel_turn(setup.base.run, turn_id, db=session, service=restricted)["turn"]["source_access_changed"]
        # The next planner turn also omits revoked donor-derived assistant text.
        next_turn = restricted.accept(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Trim ending", idempotency_key="safe-next", actor="operator", now=NOW)
        assert "example-secret" not in json.dumps(next_turn.context_snapshot)


def test_valid_shorter_aac_coverage_does_not_reject_whole_video_asr(setup, monkeypatch):
    monkeypatch.setattr(assistant_audio, "probe_trusted_audio", lambda *args, **kwargs:
        assistant_audio.AudioMetadata(2.98, 48000, 2, "stereo"))
    with setup.factory.begin() as session:
        value = context(setup, session)
    asr = ASR(duration=2.98)
    enriched = enrich(setup, value, asr)
    assert asr.calls[0][1:3] == (0, 2.98)
    assert enriched.selected.duration_sec == 3
    assert enriched.speech_evidence[0].source_duration_sec == 2.98
    assert any("soundtrack ends" in note for note in enriched.evidence_limitations)
