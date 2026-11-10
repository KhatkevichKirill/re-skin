"""Trusted admission in isolated SQLite/private media fixtures; no model calls."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import uuid

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

from app import assistant_media, assistant_service
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_contracts import EditPlan, PlanSourceError
from app.assistant_jobs import AssistantJob, AssistantJobOutbox, accept_job, request_cancel
from app.assistant_models import AssistantMessage, AssistantSelection, AssistantVersion
from app.assistant_service import (
    AssistantAdmissionConflict, AssistantAdmissionDisabled, AssistantAdmissionError,
    AssistantAdmissionService, AssistantCapabilityError,
)
from app.assistant_store import (
    ArtifactInput, ArtifactIntegrityError, AssistantArtifactStorage, AssistantConflict, get_version_artifact,
    import_run_version, publish_version, reserve_version,
)
from app.db import Base
from app.models import Run, VideoProject

NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
MEDIA = assistant_media.MediaMetadata(3, 3, 160, 120, "30", "1:1", True)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'admission.db'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    data = tmp_path / "data"
    data.mkdir()
    storage = AssistantArtifactStorage(data)
    sources = []
    with factory.begin() as session:
        for index in range(3):
            source = data / f"source-{index}.mp4"
            source.write_bytes(f"fixture-{index}".encode())
            project = VideoProject(source_type="upload", source_ref="fixture")
            session.add(project)
            session.flush()
            # Deliberately wrong legacy metadata: admission must use its probe.
            run = Run(project_id=project.id, status="done", result_local_path=str(source), result_duration_sec=20)
            session.add(run)
            session.flush()
            version = import_run_version(session, run, actor="operator", storage=storage)
            artifact = get_version_artifact(session, version.id)
            sources.append(SimpleNamespace(
                project=project.id, run=run.id, version=version.id, artifact=artifact.id,
                conversation=version.conversation_id, path=source,
            ))
    probes = []
    probe_metadata = {}

    def probe(path, **kwargs):
        probes.append(Path(path))
        return probe_metadata.get(Path(path).read_bytes(), MEDIA)

    monkeypatch.setattr(assistant_media, "probe_trusted_media", probe)

    def service(*, policy=None, enabled=True):
        return AssistantAdmissionService(
            storage=storage, policy=ProjectAllowlist([sources[0].project, sources[1].project])
            if policy is None else policy, enabled=enabled,
        )

    yield SimpleNamespace(
        factory=factory, storage=storage, data=data, base=sources[0], donor=sources[1], denied=sources[2],
        service=service, probes=probes, probe_metadata=probe_metadata, original_probe=probe,
    )
    engine.dispose()


def plan(context, *, base=None, operations=None):
    return EditPlan.model_validate({
        "run_id": context.base.run, "base_version_id": base or context.base.version,
        "operations": operations or [{"kind": "trim", "start_sec": 0, "end_sec": 2}],
    })


def admit(context, session, **kwargs):
    defaults = dict(run_id=context.base.run, plan=plan(context), instruction="Trim ending",
                    idempotency_key="request-1", actor="operator", now=NOW)
    defaults.update(kwargs)
    return context.service().admit(session, **defaults)


def counts(session):
    return tuple(session.scalar(select(func.count()).select_from(model))
                 for model in (AssistantJob, AssistantMessage, AssistantJobOutbox, AssistantVersion))


def child(context, session):
    version = reserve_version(session, plan(context), idempotency_key="other-edit", actor="operator")
    artifact = ArtifactInput(context.base.path, metadata={"duration_sec": 3})
    publish_version(session, version.id, full=artifact, preview=artifact, storage=context.storage)
    return version.id


def replacement(context, donor=None, **extra):
    donor = donor or context.donor
    return plan(context, operations=[{
        "kind": "replace_footage", "start_sec": 1, "end_sec": 2,
        "source": {"asset_id": donor.artifact, "version_id": donor.version}, **extra,
    }])


def test_default_disabled_and_invalid_truthy_enable_flags_do_not_probe_or_accept(setup):
    with setup.factory() as session:
        for enabled in (False, None, "true", 1):
            with pytest.raises(AssistantAdmissionDisabled):
                setup.service(enabled=enabled).admit(
                    session, run_id=setup.base.run, plan=plan(setup), instruction="trim",
                    idempotency_key="request", actor="operator", now=NOW,
                )
        service = AssistantAdmissionService(storage=setup.storage, policy=ProjectAllowlist([setup.base.project]))
        with pytest.raises(AssistantAdmissionDisabled):
            service.pin_base(session, run_id=setup.base.run)
        assert counts(session) == (0, 0, 0, 3) and setup.probes == []


def test_admission_pins_private_bytes_and_records_probe_not_stored_metadata(setup):
    with setup.factory.begin() as session:
        receipt = admit(setup, session)
        message = session.get(AssistantMessage, receipt.message_id)
        snapshot = message.context_snapshot
        assert snapshot["base"]["media"]["video_duration_sec"] == 3
        assert snapshot["base"]["media"]["duration_sec"] == 3
        assert snapshot["base_version_id"] == setup.base.version
        assert snapshot["job_id"] == receipt.job_id
        assert "storage_key" not in str(snapshot) and str(setup.data) not in str(snapshot)
        assert all(path.is_relative_to(setup.storage.root) for path in setup.probes)
        assert counts(session) == (1, 1, 1, 4)
        assert setup.service().revalidate_job(session, receipt.job_id) == receipt


@pytest.mark.parametrize("operation", [
    {"kind": "cut", "start_sec": 1, "end_sec": 2},
    {"kind": "image_fade", "start_sec": 1, "end_sec": 3},
])
def test_supported_internal_cut_and_ending_fade_are_admitted(setup, operation):
    with setup.factory.begin() as session:
        receipt = admit(setup, session, plan=plan(setup, operations=[operation]))
        assert session.get(AssistantJob, receipt.job_id).accepted_plan["operations"][0]["kind"] == operation["kind"]


def test_caller_rollback_removes_job_message_version_and_outbox(setup):
    with setup.factory() as session:
        admit(setup, session)
        session.rollback()
    with setup.factory() as session:
        assert counts(session) == (0, 0, 0, 3)


def test_message_failure_rolls_back_job_version_and_outbox_together(setup, monkeypatch):
    def fail(*args, **kwargs):
        raise AssistantAdmissionError("injected message failure")

    monkeypatch.setattr(assistant_service, "append_message", fail)
    with setup.factory.begin() as session:
        with pytest.raises(AssistantAdmissionError):
            admit(setup, session)
        assert counts(session) == (0, 0, 0, 3)


def test_retry_keeps_original_base_and_snapshot_after_selection_changes(setup):
    with setup.factory.begin() as session:
        first = admit(setup, session)
        snapshot = session.get(AssistantMessage, first.message_id).context_snapshot
        new_base = child(setup, session)
        assert setup.service().pin_base(session, run_id=setup.base.run) == new_base
        second = admit(setup, session)
        assert second == first and second.base_version_id == setup.base.version
        assert session.get(AssistantMessage, second.message_id).context_snapshot == snapshot
        assert counts(session) == (1, 1, 1, 5)


@pytest.mark.parametrize("change", ["instruction", "actor", "plan"])
def test_same_request_key_conflicts_on_changed_instruction_actor_or_plan(setup, change):
    with setup.factory.begin() as session:
        first = admit(setup, session)
        kwargs = {
            "instruction": {"instruction": "Another request"},
            "actor": {"actor": "different-operator"},
            "plan": {"plan": plan(setup, operations=[{"kind": "trim", "start_sec": 0, "end_sec": 1}])},
        }[change]
        with pytest.raises((AssistantAdmissionConflict, AssistantConflict)):
            admit(setup, session, **kwargs)
        assert counts(session) == (1, 1, 1, 4)
        assert session.get(AssistantJob, first.job_id).pinned_base_version_id == setup.base.version


def test_explicit_earlier_base_is_valid_but_selected_default_does_not_silently_rebase(setup):
    with setup.factory.begin() as session:
        child(setup, session)
        with pytest.raises(AssistantAdmissionConflict):
            admit(setup, session)
        receipt = admit(setup, session, explicit_base_version_id=setup.base.version)
        assert receipt.base_version_id == setup.base.version


def test_selection_changed_while_probing_is_rechecked_before_acceptance(setup, monkeypatch):
    with setup.factory.begin() as session:
        new_base = child(setup, session)
        selection = session.get(AssistantSelection, setup.base.conversation)
        # Set up the selected original before the request, then change it during probe.
        selection.selected_version_id = setup.base.version
        session.flush()

        def probe(path, **kwargs):
            selection.selected_version_id = new_base
            session.flush()
            return MEDIA

        monkeypatch.setattr(assistant_media, "probe_trusted_media", probe)
        with pytest.raises(AssistantAdmissionConflict, match="selected base changed"):
            admit(setup, session)
        assert counts(session) == (0, 0, 0, 4)


def test_base_cannot_come_from_another_run_even_inside_allowed_scope(setup):
    with setup.factory() as session:
        with pytest.raises(AssistantAdmissionConflict):
            admit(setup, session, plan=plan(setup, base=setup.donor.version), explicit_base_version_id=setup.donor.version)
        assert counts(session) == (0, 0, 0, 3)


def test_cross_project_donor_is_allowed_only_when_target_and_donor_are_allowlisted(setup):
    with setup.factory.begin() as session:
        receipt = admit(setup, session, plan=replacement(setup))
        assert setup.service().revalidate_job(session, receipt.job_id) == receipt
        with pytest.raises(AssistantAccessDenied):
            setup.service(policy=ProjectAllowlist([setup.base.project])).revalidate_job(session, receipt.job_id)
        with pytest.raises(AssistantAccessDenied):
            setup.service(policy=ProjectAllowlist([setup.base.project])).admit(
                session, run_id=setup.base.run, plan=replacement(setup), instruction="Trim ending",
                idempotency_key="request-1", actor="operator", now=NOW,
            )
        with pytest.raises(AssistantAccessDenied):
            admit(setup, session, idempotency_key="denied", plan=replacement(setup, setup.denied))
        with pytest.raises(AssistantAccessDenied):
            setup.service(policy=ProjectAllowlist()).revalidate_job(session, receipt.job_id)


@pytest.mark.parametrize("missing", ["base", "donor"])
def test_unavailable_pinned_source_is_denied_at_admission_and_worker_recheck(setup, missing):
    with setup.factory.begin() as session:
        receipt = admit(setup, session, plan=replacement(setup))
        source = setup.base if missing == "base" else setup.donor
        artifact = get_version_artifact(session, source.version)
        setup.storage.path_for(artifact.storage_key).unlink()
        with pytest.raises(AssistantAccessDenied):
            setup.service().revalidate_job(session, receipt.job_id)
        with pytest.raises(AssistantAccessDenied):
            admit(setup, session, idempotency_key="missing", plan=replacement(setup))


@pytest.mark.parametrize("operations", [
    [{"kind":"image_fade", "start_sec":0, "end_sec":1/30}],
    [{"kind":"image_fade", "start_sec":.05, "end_sec":2}],
    [{"kind": "cut", "start_sec": 0, "end_sec": 1}],
    [{"kind": "cut", "start_sec": 1, "end_sec": 3}],
    [{"kind": "trim", "start_sec": 0.05, "end_sec": 2}],
    [{"kind": "subtitle_edit", "changes": [{"action": "delete", "cue_id": "cue-1"}]}],
])
def test_unsupported_capabilities_and_nonframe_boundaries_never_create_work(setup, operations):
    with setup.factory() as session:
        with pytest.raises(AssistantCapabilityError):
            admit(setup, session, plan=plan(setup, operations=operations))
        assert counts(session) == (0, 0, 0, 3)


@pytest.mark.parametrize("kind", ["audio_splice", "full_track_audio"])
def test_audio_capability_is_admitted_with_authorized_available_donor(setup, kind):
    operation = {"kind": kind, "source": {"asset_id": setup.donor.artifact, "version_id": setup.donor.version}}
    if kind == "audio_splice":
        operation.update(start_sec=0, end_sec=1)
    with setup.factory() as session:
        receipt = admit(setup, session, plan=plan(setup, operations=[operation]))
        assert session.get(AssistantJob, receipt.job_id).accepted_plan["operations"][0]["kind"] == kind
        assert counts(session) == (1, 1, 1, 4)


def test_actual_probe_duration_overrules_legacy_duration_claim(setup):
    with setup.factory() as session:
        with pytest.raises(PlanSourceError):
            admit(setup, session, plan=plan(setup, operations=[{"kind": "trim", "start_sec": 0, "end_sec": 4}]))


def test_pinned_bytes_changed_during_probe_never_create_work(setup, monkeypatch):
    def mutate(path, **kwargs):
        path = Path(path)
        path.unlink()
        path.write_bytes(b"changed during backend probe")
        return MEDIA

    monkeypatch.setattr(assistant_media, "probe_trusted_media", mutate)
    with setup.factory() as session:
        with pytest.raises(ArtifactIntegrityError):
            admit(setup, session)
        assert counts(session) == (0, 0, 0, 3)


def test_replacement_uses_actual_donor_geometry_and_duration(setup):
    setup.probe_metadata[b"fixture-1"] = replace(MEDIA, width=320)
    with setup.factory() as session:
        with pytest.raises(AssistantCapabilityError, match="geometry"):
            admit(setup, session, plan=replacement(setup))
        setup.probe_metadata[b"fixture-1"] = replace(MEDIA, video_duration_sec=0.5)
        with pytest.raises(PlanSourceError):
            admit(setup, session, plan=replacement(setup))


def test_cancelled_job_and_job_without_admission_message_are_not_worker_authority(setup):
    with setup.factory.begin() as session:
        receipt = admit(setup, session)
        request_cancel(session, receipt.job_id, now=NOW)
        with pytest.raises(AssistantAdmissionConflict):
            setup.service().revalidate_job(session, receipt.job_id)
        raw = accept_job(session, plan(setup), request_scope="assistant-edit:" + setup.base.run,
                         idempotency_key="raw", actor="operator", now=NOW)
        with pytest.raises(AssistantAdmissionConflict, match="message"):
            setup.service().revalidate_job(session, raw.id)


@pytest.mark.parametrize("mismatch", ["conversation", "recipe"])
def test_worker_recheck_rejects_persisted_job_and_pending_version_disagreement(setup, mismatch):
    with setup.factory.begin() as session:
        receipt = admit(setup, session)
        original = session.get(AssistantJob, receipt.job_id)
        other_plan = (
            EditPlan.model_validate({
                "run_id": setup.donor.run, "base_version_id": setup.donor.version,
                "operations": [{"kind": "trim", "start_sec": 0, "end_sec": 2}],
            }) if mismatch == "conversation" else plan(setup, operations=[
                {"kind": "trim", "start_sec": 0, "end_sec": 1},
            ])
        )
        unrelated = reserve_version(session, other_plan, idempotency_key="unrelated", actor="operator")
        # Model a misbound persisted record created outside the admission path.
        # No immutable row is updated and no database guard is disabled.
        fields = {column.key: getattr(original, column.key) for column in AssistantJob.__table__.columns}
        forged = AssistantJob(**{
            **fields, "id": str(uuid.uuid4()), "idempotency_key": "misbound",
            "pending_version_id": unrelated.id,
        })
        session.add(forged)
        session.flush()
        before = len(setup.probes)
        with pytest.raises(AssistantAdmissionConflict, match="bindings"):
            setup.service().revalidate_job(session, forged.id)
        assert len(setup.probes) == before


def test_public_probe_uses_actual_normalized_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires local FFmpeg")
    source = tmp_path / "actual.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "lavfi",
        "-i", "color=c=white:size=160x120:rate=30:duration=2", "-an", "-c:v", "libx264",
        "-threads", "1", "-pix_fmt", "yuv420p", str(source),
    ], check=True, timeout=20, capture_output=True)
    metadata = assistant_media.probe_trusted_media(source)
    assert metadata.video_duration_sec == 2 and metadata.fps == "30" and metadata.has_audio is False


def test_contiguous_dip_admitted_and_revalidated_as_one_job(setup):
    operations = [{'kind':'image_fade', 'start_sec':.5, 'end_sec':1.5, 'direction':'out'},
                  {'kind':'image_fade', 'start_sec':1.5, 'end_sec':2.5, 'direction':'in'}]
    with setup.factory.begin() as session:
        receipt = admit(setup, session, plan=plan(setup, operations=operations))
        assert session.get(AssistantJob, receipt.job_id).accepted_plan['operations'] == operations
        assert setup.service().revalidate_job(session, receipt.job_id) == receipt
        assert counts(session) == (1, 1, 1, 4)


@pytest.mark.parametrize('directions,gap', [('inout',0), ('outout',0), ('outin',.1)])
def test_separate_fade_shapes_are_supported(setup, directions, gap):
    first, second = ('out','in') if directions == 'outin' else (('in','out') if directions == 'inout' else ('out','out'))
    operations = [{'kind':'image_fade', 'start_sec':0, 'end_sec':1, 'direction':first},
                  {'kind':'image_fade', 'start_sec':1+gap, 'end_sec':2, 'direction':second}]
    with setup.factory.begin() as session:
        receipt = admit(setup, session, plan=plan(setup, operations=operations))
        assert setup.service().revalidate_job(session, receipt.job_id) == receipt


def test_repeated_donor_is_probed_once_and_later_operation_checked(setup):
    donor = {'asset_id':setup.donor.artifact,'version_id':setup.donor.version}
    with setup.factory.begin() as session:
        receipt = admit(setup,session,plan=plan(setup,operations=[
            {'kind':'audio_splice','start_sec':0,'end_sec':1,'source':donor},
            {'kind':'audio_splice','start_sec':2,'end_sec':3,'source':donor}]))
        assert len(setup.probes) == 2
        assert len(session.get(AssistantJob,receipt.job_id).accepted_plan['operations']) == 2


def test_later_replacement_must_pass_donor_geometry_validation(setup):
    setup.probe_metadata[setup.donor.path.read_bytes()] = replace(MEDIA,width=320)
    with setup.factory.begin() as session:
        with pytest.raises(AssistantCapabilityError,match='geometry'):
            admit(setup,session,plan=plan(setup,operations=[
                {'kind':'image_fade','start_sec':0,'end_sec':1},
                {'kind':'replace_footage','start_sec':1,'end_sec':2,
                 'source':{'asset_id':setup.donor.artifact,'version_id':setup.donor.version}}]))
        assert counts(session) == (0,0,0,3)
