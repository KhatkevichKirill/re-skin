"""Disabled-by-default trusted admission for the staged local-edit capability.

No endpoint, model/provider call, render, queue dispatch, or transaction commit.
Planning callers must pin a base before planning and pass that exact explicit
base at admission. The selected-base convenience path rejects selection drift;
retries always use the original accepted base. Returned data contains no paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
import hashlib
import math
import re
from collections.abc import Mapping

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .assistant_access import AssistantAccessDenied, AuthorizedArtifactCatalog, ProjectAllowlist, canonical_uuid
from .assistant_contracts import (
    AssetManifest, CutOperation, EditPlan, ImageFadeOperation, ReplaceFootageOperation,
    TrimOperation, VersionManifest, validate_plan_sources, validate_local_plan_capability,
    AudioSpliceOperation, FullTrackAudioOperation, SubtitleEditOperation, ExtendEndingOperation,
)
from . import assistant_media
from . import assistant_audio, assistant_subtitles
from .assistant_jobs import AssistantJob, _atomic, accept_job
from .assistant_models import (
    AssistantArtifact, AssistantConversation, AssistantMessage, AssistantSelection, AssistantVersion,
)
from .assistant_store import AssistantArtifactStorage, append_message, get_version_artifact, verify_artifact
from .models import Run, VideoProject

_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


class AssistantAdmissionError(ValueError):
    pass


class AssistantAdmissionDisabled(AssistantAdmissionError):
    pass


class AssistantCapabilityError(AssistantAdmissionError):
    pass


class AssistantAdmissionConflict(AssistantAdmissionError):
    pass


@dataclass(frozen=True)
class AdmissionReceipt:
    job_id: str
    pending_version_id: str
    base_version_id: str
    message_id: str


def _key(value: object) -> str:
    if not isinstance(value, str) or _KEY.fullmatch(value) is None:
        raise AssistantAdmissionError("expected a bounded ASCII request identifier")
    return value


def _frame_boundary(value: float, media: assistant_media.MediaMetadata) -> None:
    frames = value * float(Fraction(media.fps))
    if abs(frames - round(frames)) > 0.0001:
        raise AssistantCapabilityError("local edit boundaries must align to video frames")


def validate_plan_semantics(
    plan: EditPlan, *, base: VersionManifest,
    base_media: assistant_media.MediaMetadata,
    assets: Mapping[str, AssetManifest],
    asset_media: Mapping[str, assistant_media.MediaMetadata],
    ending_audio: Mapping[str, assistant_audio.AudioMetadata] | None = None,
) -> None:
    """Validate the staged adapter subset against trusted backend probes only."""
    validate_plan_sources(plan, base_version=base, assets=assets)
    try:
        validate_local_plan_capability(plan, duration_sec=base_media.video_duration_sec, fps=base_media.fps)
    except ValueError as exc:
        raise AssistantCapabilityError(str(exc)) from exc
    for operation in plan.operations:
        if isinstance(operation, ExtendEndingOperation):
            from .assistant_ending import resolve_ending_timing
            donor_audio = (ending_audio or {}).get(operation.source.asset_id)
            if donor_audio is None:
                raise AssistantCapabilityError('ending donor requires verified audio coverage')
            try:
                resolve_ending_timing(base_media,start_sec=operation.start_sec,end_sec=operation.end_sec,
                    donor_start_sec=operation.donor_start_sec,visual_kind=operation.visual.kind,
                    freeze_timestamp_sec=getattr(operation.visual,'timestamp_sec',None),fade_out_sec=operation.fade_out_sec,
                    donor_audio_duration_sec=donor_audio.duration_sec)
            except assistant_media.AssistantMediaError as error:
                raise AssistantCapabilityError(str(error)) from error
        if isinstance(operation, (AudioSpliceOperation, FullTrackAudioOperation)) and not base_media.has_audio:
            raise AssistantCapabilityError("audio replacement requires a base soundtrack")
        if isinstance(operation, ReplaceFootageOperation):
            donor = asset_media[operation.source.asset_id]
            if (base_media.width, base_media.height, base_media.fps, base_media.sample_aspect_ratio) != (
                donor.width, donor.height, donor.fps, donor.sample_aspect_ratio,
            ):
                raise AssistantCapabilityError("replacement geometry, FPS and SAR must match the base")
            _frame_boundary(operation.donor_start_sec, donor)
            _frame_boundary(operation.donor_start_sec + operation.end_sec - operation.start_sec, donor)


class AssistantAdmissionService:
    def __init__(
        self, *, storage: AssistantArtifactStorage, policy: ProjectAllowlist,
        enabled: bool = False, probe_timeout_sec: float = 30,
    ):
        if (type(probe_timeout_sec) not in (int, float) or not math.isfinite(probe_timeout_sec)
                or not 0 < probe_timeout_sec <= 120):
            raise AssistantAdmissionError("probe timeout must be within (0,120] seconds")
        self._storage, self._policy = storage, policy
        self._enabled, self._probe_timeout = enabled is True, probe_timeout_sec

    def _gate(self):
        if not self._enabled:
            raise AssistantAdmissionDisabled("assistant admission is disabled")

    @staticmethod
    def _scope(run_id: str) -> str:
        return "assistant-edit:" + run_id

    @staticmethod
    def _message_key(scope: str, key: str) -> str:
        return "admission:" + hashlib.sha256((scope + "\0" + key).encode()).hexdigest()

    def _ending_proof(self, session, plan, turn_id):
        if not any(isinstance(operation,ExtendEndingOperation) for operation in plan.operations):
            return None
        if turn_id is None:
            raise AssistantCapabilityError('ending extension requires a verified agent candidate')
        from .assistant_turns import AssistantTurnService
        turns = AssistantTurnService(storage=self._storage,policy=self._policy,enabled=True)
        turn = turns.require_turn(session,plan.run_id,turn_id)
        if turn.status != 'succeeded':
            raise AssistantCapabilityError('ending extension requires a completed agent proposal')
        turns.require_verified_ending(session,turn,plan)
        return turn.id

    def _conversation(self, session: Session, run_id: str) -> AssistantConversation:
        conversation = session.scalar(select(AssistantConversation).join(
            Run, (Run.id == AssistantConversation.run_id) & (Run.project_id == AssistantConversation.project_id),
        ).join(VideoProject, VideoProject.id == AssistantConversation.project_id).where(
            AssistantConversation.run_id == run_id,
            self._policy.sql_filter(AssistantConversation.project_id),
        ).execution_options(populate_existing=True, autoflush=False))
        if conversation is None:
            raise AssistantAccessDenied("assistant access denied")
        self._policy.require_project(conversation.project_id)
        return conversation

    def pin_base(
        self, session: Session, *, run_id: str, explicit_base_version_id: str | None = None,
    ) -> str:
        """Resolve once before planning; persist/pass this exact base afterward."""
        self._gate()
        run_id = canonical_uuid(run_id)
        conversation = self._conversation(session, run_id)
        if explicit_base_version_id is None:
            selection = session.get(AssistantSelection, conversation.id, populate_existing=True)
            version_id = selection.selected_version_id if selection else None
            if version_id is None:
                raise AssistantAdmissionConflict("Run has no selected successful base")
        else:
            version_id = canonical_uuid(explicit_base_version_id)
        # No artifact probe is needed just to choose a base, but scope/status is.
        catalog = AuthorizedArtifactCatalog(session, policy=self._policy, is_available=lambda artifact: False)
        target = catalog.require_editable_target(version_id)
        if target.run_id != run_id:
            raise AssistantAdmissionConflict("base belongs to another Run")
        return target.version_id

    def _validate(self, session: Session, plan: EditPlan, *, cancelled=None, timeout_sec=120) -> dict:
        paths = {}
        budget = assistant_media._budget(timeout_sec,cancelled)
        def check():
            budget.check()
            return False
        def verify(artifact):
            return verify_artifact(artifact,storage=self._storage,**({"cancelled":check} if cancelled else {}))
        def remaining():
            budget.check()
            import time
            return min(self._probe_timeout,budget.deadline-time.monotonic())

        def available(artifact):
            paths[artifact.id] = verify(artifact)
            return True

        catalog = AuthorizedArtifactCatalog(session, policy=self._policy, is_available=available)
        target = catalog.require_editable_target(plan.base_version_id)
        if target.run_id != plan.run_id:
            raise AssistantAdmissionConflict("plan base belongs to another Run")
        base_artifact = get_version_artifact(session, plan.base_version_id)
        base_descriptor = catalog.resolve_editable_reference(
            target_version_id=plan.base_version_id, asset_id=base_artifact.id,
            version_id=plan.base_version_id,
        )
        if base_descriptor.media_kind != "video":
            raise AssistantCapabilityError("local base must be a video")
        try:
            validate_local_plan_capability(plan)
        except ValueError as exc:
            raise AssistantCapabilityError(str(exc)) from exc
        descriptors, media_by_id, ending_audio = {}, {}, {}

        def probe(descriptor):
            probe_fn = assistant_audio.probe_trusted_audio if descriptor.media_kind == "audio" else assistant_media.probe_trusted_media
            media = probe_fn(
                paths[descriptor.asset_id], timeout_sec=remaining(), cancelled=check,
            )
            artifact = session.get(AssistantArtifact, descriptor.asset_id, populate_existing=True)
            verify(artifact)  # Detect mutation during probe.
            return media

        base_media = probe(base_descriptor)
        for operation in plan.operations:
            source = getattr(operation, "source", None)
            if source is not None and source.asset_id not in descriptors:
                descriptor = catalog.resolve_editable_reference(
                    target_version_id=plan.base_version_id, asset_id=source.asset_id,
                    version_id=source.version_id,
                )
                if descriptor.media_kind not in ("video", "audio") or (
                    isinstance(operation, ReplaceFootageOperation) and descriptor.media_kind != "video"
                ):
                    raise AssistantCapabilityError("unsupported donor media kind")
                descriptors[source.asset_id] = descriptor
                media_by_id[source.asset_id] = probe(descriptor)
            if isinstance(operation, ExtendEndingOperation):
                ending_audio[source.asset_id] = assistant_audio.probe_trusted_audio(paths[source.asset_id],
                    timeout_sec=remaining(),cancelled=check)
                verify(session.get(AssistantArtifact,source.asset_id,populate_existing=True))
        track = assistant_subtitles.get_subtitle_context(session, plan.base_version_id, storage=self._storage, **({"cancelled":check} if cancelled is not None else {}))
        if track is not None:
            clean_artifact = get_version_artifact(session, plan.base_version_id, role="clean")
            clean_media = assistant_media.probe_trusted_media(
                verify(clean_artifact), timeout_sec=remaining(), cancelled=check,
            )
            if clean_media != base_media:
                raise AssistantCapabilityError("clean subtitle picture must match base media timing and geometry")
            assistant_subtitles.validate_duration(track, base_media.video_duration_sec)
        for operation in plan.operations:
            if isinstance(operation, SubtitleEditOperation):
                if track is None:
                    raise AssistantCapabilityError("subtitle edit requires editable cues and verified clean source")
                assistant_subtitles.edit_track(track, operation, duration_sec=base_media.video_duration_sec)
        base = VersionManifest(
            version_id=plan.base_version_id, run_id=plan.run_id,
            duration_sec=base_media.video_duration_sec, status="succeeded", available=True,
            editable_subtitle_cue_ids=tuple(c.cue_id for c in track.cues) if track else (),
        )
        assets = {asset_id: AssetManifest(
            asset_id=asset_id, version_id=descriptor.version_id, media_kind=descriptor.media_kind,
            duration_sec=(media_by_id[asset_id].duration_sec if descriptor.media_kind == "audio" else media_by_id[asset_id].video_duration_sec),
            has_audio=(True if descriptor.media_kind == "audio" else media_by_id[asset_id].has_audio), available=True,
        ) for asset_id, descriptor in descriptors.items()}
        validate_plan_semantics(plan, base=base, base_media=base_media, assets=assets, asset_media=media_by_id,
            ending_audio=ending_audio)

        def snapshot(descriptor, media):
            # Discard stored probe claims; only this backend probe is authoritative.
            return {
                "asset_id": descriptor.asset_id, "version_id": descriptor.version_id,
                "project_id": descriptor.project_id, "sha256": descriptor.sha256,
                "size_bytes": descriptor.size_bytes, "media": asdict(media),
            }

        return {
            "schema_version": 1, "run_id": plan.run_id, "base_version_id": plan.base_version_id,
            "plan": plan.model_dump(mode="json"), "base": snapshot(base_descriptor, base_media),
            "donors": [snapshot(descriptors[key], media_by_id[key]) for key in sorted(descriptors)],
        }

    @staticmethod
    def _existing(session: Session, scope: str, key: str):
        return session.scalar(select(AssistantJob).where(
            AssistantJob.request_scope == scope, AssistantJob.idempotency_key == key,
        ).execution_options(populate_existing=True, autoflush=False))

    def admit(
        self, session: Session, *, run_id: str, plan: EditPlan, instruction: str,
        idempotency_key: str, actor: str, now: datetime,
        explicit_base_version_id: str | None = None,
        verified_candidate_turn_id: str | None = None,
    ) -> AdmissionReceipt:
        self._gate()
        run_id, key, actor = canonical_uuid(run_id), _key(idempotency_key), _key(actor)
        plan = EditPlan.model_validate(plan.model_dump(mode="json"))
        if plan.run_id != run_id:
            raise AssistantAdmissionConflict("plan Run does not match the requested Run")
        verified_candidate_turn_id = self._ending_proof(session,plan,verified_candidate_turn_id)
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction.encode()) > 256 * 1024:
            raise AssistantAdmissionError("instruction must be nonempty and at most 256 KiB")
        conversation = self._conversation(session, run_id)
        scope = self._scope(run_id)
        existing = self._existing(session, scope, key)
        if existing is not None:
            expected = existing.pinned_base_version_id
            if existing.actor != actor or existing.accepted_plan != plan.model_dump(mode="json"):
                raise AssistantAdmissionConflict("request key was already used for another plan or actor")
        else:
            expected = self.pin_base(session, run_id=run_id, explicit_base_version_id=explicit_base_version_id)
        if (plan.base_version_id != expected or (explicit_base_version_id is not None
                and canonical_uuid(explicit_base_version_id) != expected)):
            raise AssistantAdmissionConflict("plan does not match the pinned request base")
        snapshot = self._validate(session, plan)
        if verified_candidate_turn_id is not None:
            snapshot['verified_candidate_turn_id'] = verified_candidate_turn_id
        message_key = self._message_key(scope, key)
        # _atomic has the SQLite real-BEGIN guard: releasing a nested savepoint
        # must not commit job/message rows if the outer caller later rolls back.
        with _atomic(session):
            session.execute(update(AssistantConversation).where(
                AssistantConversation.id == conversation.id,
            ).values(next_version_sequence=AssistantConversation.next_version_sequence))
            current = self._existing(session, scope, key)
            if current is None:
                from .assistant_run_result import ensure_no_unresolved_intent
                ensure_no_unresolved_intent(session, run_id)
            if current is None and explicit_base_version_id is None:
                if self.pin_base(session, run_id=run_id) != plan.base_version_id:
                    raise AssistantAdmissionConflict("selected base changed before acceptance")
            job = accept_job(session, plan, request_scope=scope, idempotency_key=key, actor=actor, now=now)
            old_message = session.scalar(select(AssistantMessage).where(
                AssistantMessage.conversation_id == conversation.id,
                AssistantMessage.idempotency_key == message_key,
            ))
            if old_message is not None:
                snapshot = old_message.context_snapshot
                if verified_candidate_turn_id is not None and snapshot.get('verified_candidate_turn_id') != verified_candidate_turn_id:
                    raise AssistantAdmissionConflict('ending admission candidate binding changed')
            else:
                snapshot = {**snapshot, "job_id": job.id}
            message = append_message(
                session, conversation.id, idempotency_key=message_key, actor=actor,
                role="operator", content=instruction, base_version_id=plan.base_version_id,
                context_snapshot=snapshot,
            )
            return AdmissionReceipt(job.id, job.pending_version_id, job.pinned_base_version_id, message.id)

    def revalidate_job(self, session: Session, job_id: str, *, cancelled=None, timeout_sec=120) -> AdmissionReceipt:
        """Recheck policy, bytes, capabilities and immutable bindings before work.

        This grants no lease or publication authority; workers still need the
        existing fenced lifecycle checks at every side-effect boundary.
        """
        self._gate()
        job = session.get(AssistantJob, canonical_uuid(job_id), populate_existing=True)
        if job is None or job.status not in ("queued", "running") or job.cancel_requested_at is not None:
            raise AssistantAdmissionConflict("job is not eligible for execution")
        plan = EditPlan.model_validate(job.accepted_plan)
        conversation = self._conversation(session, plan.run_id)
        pending = session.get(AssistantVersion, job.pending_version_id, populate_existing=True)
        if (job.request_scope != self._scope(plan.run_id) or job.pinned_base_version_id != plan.base_version_id
                or pending is None or pending.status != "pending"
                or pending.conversation_id != conversation.id
                or pending.parent_version_id != plan.base_version_id or pending.base_version_id != plan.base_version_id
                or pending.actor != job.actor or pending.recipe != plan.model_dump(mode="json")):
            raise AssistantAdmissionConflict("persisted job/version bindings disagree")
        if any(isinstance(operation,ExtendEndingOperation) for operation in plan.operations):
            proof_message = session.scalar(select(AssistantMessage).where(
                AssistantMessage.conversation_id == conversation.id,
                AssistantMessage.idempotency_key == self._message_key(job.request_scope,job.idempotency_key)))
            if (proof_message is None or proof_message.actor != job.actor or proof_message.role != 'operator'
                    or proof_message.base_version_id != plan.base_version_id or not isinstance(proof_message.context_snapshot,dict)
                    or proof_message.context_snapshot.get('job_id') != job.id
                    or proof_message.context_snapshot.get('plan') != plan.model_dump(mode='json')):
                raise AssistantAdmissionConflict('ending admission proof is missing')
            self._ending_proof(session,plan,proof_message.context_snapshot.get('verified_candidate_turn_id'))
        self._validate(session, plan, cancelled=cancelled, timeout_sec=timeout_sec)
        message = session.scalar(select(AssistantMessage).where(
            AssistantMessage.conversation_id == conversation.id,
            AssistantMessage.idempotency_key == self._message_key(job.request_scope, job.idempotency_key),
        ))
        if (message is None or message.actor != job.actor or message.role != "operator"
                or message.base_version_id != plan.base_version_id
                or not isinstance(message.context_snapshot, dict)
                or message.context_snapshot.get("job_id") != job.id
                or message.context_snapshot.get("plan") != plan.model_dump(mode="json")):
            raise AssistantAdmissionConflict("persisted admission message disagrees with job")
        return AdmissionReceipt(job.id, job.pending_version_id, job.pinned_base_version_id, message.id)
