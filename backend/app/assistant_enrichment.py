"""Optional bounded offline speech evidence for a pinned planning turn.

The trusted supervisor calls the credential-free ASR sidecar. Recognition gives
candidate word windows, never pronunciation/speaker or visual verification.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import time

from sqlalchemy import case, func, select

from .assistant_access import AssistantAccessDenied
from .assistant_asr import transcribe_authorized_artifact
from .assistant_broker_client import context_fingerprint
from .assistant_jobs import LostLease
from .assistant_models import AssistantArtifact, AssistantConversation, AssistantVersion, AssistantVersionArtifact
from .assistant_planner import PlannerContext, PlannerSource, PlannerSpeechEvidence, PlannerSpeechWord
from .assistant_store import verify_artifact
from . import assistant_audio, assistant_media

MAX_SPEECH_SOURCE_BYTES = 128 * 1024 * 1024
_LOCALIZATION = re.compile(r"audio|sound|voice|speech|pronounc|word|phrase|subtitl|caption|dub|"
    r"аудио|звук|голос|реч|произнош|слов|фраз|субтит|音声|字幕|発音|pronunci|subt[ií]t|palabr|legenda|tonspur", re.IGNORECASE)


@dataclass
class _Budget:
    deadline: float
    cancelled: object
    monotonic: object

    def check(self):
        if self.cancelled and self.cancelled():
            raise assistant_media.AssistantMediaCancelled("speech evidence cancelled")
        if self.monotonic() >= self.deadline:
            raise assistant_media.AssistantMediaError("speech evidence budget expired")
        return False

    def remaining(self):
        self.check()
        return min(90, max(0.1, self.deadline - self.monotonic()))


class AssistantContextEnricher:
    def __init__(self, session_factory, *, service, asr, timeout_sec: int = 90, monotonic=time.monotonic):
        if type(timeout_sec) is not int or not 1 <= timeout_sec <= 90:
            raise ValueError("speech enrichment deadline must be within [1,90] seconds")
        self.factory, self.service, self.asr = session_factory, service, asr
        self.timeout_sec, self.monotonic = timeout_sec, monotonic

    @staticmethod
    def requested(context: PlannerContext) -> bool:
        return _LOCALIZATION.search(context.dialogue[-1].text) is not None

    def _probe_source(self, session, descriptor, budget):
        artifact = session.get(AssistantArtifact, descriptor.asset_id, populate_existing=True)
        if artifact is None or artifact.size_bytes > MAX_SPEECH_SOURCE_BYTES:
            raise assistant_media.AssistantMediaError("speech source exceeds evidence size limit")
        path = verify_artifact(artifact, storage=self.service.storage)
        budget.check()
        if descriptor.media_kind == "video":
            info = assistant_media.probe_trusted_media(path, timeout_sec=budget.remaining(), cancelled=budget.check)
            duration, audio, fps = info.video_duration_sec, info.has_audio, info.fps
        else:
            info = assistant_audio.probe_trusted_audio(path, timeout_sec=budget.remaining(), cancelled=budget.check)
            duration, audio, fps = info.duration_sec, True, None
        verify_artifact(artifact, storage=self.service.storage)
        budget.check()
        return PlannerSource(asset_id=descriptor.asset_id, version_id=descriptor.version_id,
            run_id=descriptor.run_id, project_id=descriptor.project_id, media_kind=descriptor.media_kind,
            duration_sec=duration, has_audio=audio, fps=fps, label=f"Run {descriptor.run_id[:8]} result")

    def _donors(self, context, budget, limitations):
        """Discover full results, using ownership joins before touching bytes."""
        from .models import Run, VideoProject
        supplied = [source for source in context.sources if source.has_audio][:2]
        if len(supplied) == 2:
            return supplied
        with self.factory() as session:
            self.service.reauthorize_context(session, context)
            # Scalar, bounded discovery does not hash every historical preview.
            rows = session.execute(select(AssistantArtifact.id, AssistantArtifact.owner_version_id,
                AssistantArtifact.size_bytes, Run.id, Run.project_id, Run.language).join(
                AssistantVersion, AssistantVersion.id == AssistantArtifact.owner_version_id).join(
                AssistantConversation, AssistantConversation.id == AssistantVersion.conversation_id).join(
                Run, (Run.id == AssistantConversation.run_id) & (Run.project_id == AssistantConversation.project_id)).join(
                VideoProject, VideoProject.id == Run.project_id).join(AssistantVersionArtifact,
                (AssistantVersionArtifact.version_id == AssistantVersion.id) & (AssistantVersionArtifact.artifact_id == AssistantArtifact.id)
                & (AssistantVersionArtifact.role == "full")).where(
                    AssistantVersion.status == "succeeded", self.service.policy.sql_filter(Run.project_id),
                    AssistantArtifact.media_kind.in_(("video", "audio")), AssistantArtifact.size_bytes <= MAX_SPEECH_SOURCE_BYTES,
                    AssistantArtifact.id != context.selected.asset_id).order_by(AssistantArtifact.id).limit(100)).all()
            language = context.settings.language.lower() if context.settings and context.settings.language else None
            rows.sort(key=lambda row: (row[3] == context.run_id,
                language is not None and (row[5] or "").lower() != language,
                row[4] != context.selected.project_id, row[0]))
            catalog = self.service.catalog(session)
            attempted = 0
            existing = {source.asset_id for source in (context.selected, *supplied)}
            for asset_id, version_id, *_ in rows:
                budget.check()
                if len(supplied) == 2 or attempted == 6:
                    break
                if asset_id in existing:
                    continue
                from .assistant_preparation import require_editable_artifact, PreparationCapabilityError
                try:
                    require_editable_artifact(session, session.get(AssistantArtifact, asset_id))
                except PreparationCapabilityError:
                    continue
                attempted += 1
                try:
                    descriptor = catalog.resolve_editable_reference(target_version_id=context.base_version_id,
                        asset_id=asset_id, version_id=version_id)
                    source = self._probe_source(session, descriptor, budget)
                    if source.has_audio:
                        supplied.append(source)
                        existing.add(source.asset_id)
                except (AssistantAccessDenied, LostLease):
                    raise
                except (ValueError, RuntimeError, OSError):
                    budget.check()
                    limitations.append("An authorized donor could not be analyzed within the supported media profile.")
            if len(rows) >= 100 or attempted == 6:
                limitations.append("Donor discovery is partial; additional authorized results were not analyzed.")
        return supplied

    def _import_donors(self, context, budget, limitations):
        from .assistant_legacy import import_legacy_run
        from .models import Run, VideoProject
        language = context.settings.language.lower() if context.settings and context.settings.language else None
        with self.factory() as session:
            self.service.require_run(session, context.run_id)
            candidates = session.scalars(select(Run).join(VideoProject, VideoProject.id == Run.project_id)
                .outerjoin(AssistantConversation, AssistantConversation.run_id == Run.id).where(
                    self.service.policy.sql_filter(Run.project_id), Run.status == "done",
                    Run.result_local_path.is_not(None), AssistantConversation.id.is_(None)).order_by(
                        case((func.lower(Run.language) == language, 0), else_=1),
                        case((Run.project_id == context.selected.project_id, 0), else_=1), Run.id).limit(2)).all()
            ids = [run.id for run in candidates]
        for run_id in ids:
            budget.check()
            try:
                with self.factory() as session:
                    run = self.service.require_run(session, run_id)
                    path = self.service.storage._source(run.result_local_path)
                    if path.stat().st_size > MAX_SPEECH_SOURCE_BYTES:
                        limitations.append("A completed donor exceeded the 128 MiB speech import limit and was skipped.")
                        continue
                    import_legacy_run(session, run, actor="operator", storage=self.service.storage,
                        policy=self.service.policy, timeout_sec=min(30, budget.remaining()), cancelled=budget.check)
                    session.commit()
            except (AssistantAccessDenied, LostLease, assistant_media.AssistantMediaCancelled):
                raise
            except (ValueError, RuntimeError, OSError):
                if budget.cancelled and budget.cancelled():
                    raise assistant_media.AssistantMediaCancelled("speech evidence cancelled")
                limitations.append("A scoped completed donor could not be imported within the speech-analysis budget.")
                if self.monotonic() >= budget.deadline:
                    break
        if ids:
            limitations.append("Automatic import examined at most two scoped completed results; other results may remain unexamined.")

    @staticmethod
    def _evidence(source, transcript, *, remaining_words):
        # The sidecar's typed output is still validated against the requested
        # source window. Drop a word crossing the edge rather than invent time.
        if transcript.network_denied is not True or not 0 <= transcript.max_rss_kib <= 2 * 1024 * 1024:
            raise assistant_media.AssistantMediaError("speech runtime did not report bounded isolation evidence")
        words, omitted = [], False
        for word in transcript.words:
            if (word.start_sec < transcript.source_start_sec or word.end_sec > transcript.source_end_sec
                    or word.end_sec <= word.start_sec or len(words) >= min(400, remaining_words)):
                omitted = True
                continue
            words.append(PlannerSpeechWord(text=word.text, start_sec=word.start_sec,
                end_sec=word.end_sec, probability=word.probability))
        return PlannerSpeechEvidence(asset_id=source.asset_id, version_id=source.version_id,
            source_start_sec=transcript.source_start_sec, source_end_sec=transcript.source_end_sec,
            source_duration_sec=transcript.source_duration_sec, language=transcript.language,
            language_probability=transcript.language_probability, words=tuple(words), words_truncated=omitted,
            coverage_complete=transcript.source_start_sec == 0 and transcript.source_end_sec == transcript.source_duration_sec)

    def enrich(self, context: PlannerContext, *, cancelled=None) -> PlannerContext:
        context = PlannerContext.model_validate(context.model_dump(mode="json"))
        if context.speech_evidence or context.evidence_limitations or not self.requested(context):
            return context
        budget = _Budget(self.monotonic() + self.timeout_sec, cancelled, self.monotonic)
        limitations = ["Word times are ASR estimates. Pronunciation, speaker, delivery, lip sync and mixed background require listening review."]
        evidence, donors = [], []
        try:
            donors = self._donors(context, budget, limitations)
            if len(donors) < 2 and len(context.sources) < 32:
                self._import_donors(context, budget, limitations)
                donors = self._donors(context, budget, limitations)
        except (AssistantAccessDenied, LostLease, assistant_media.AssistantMediaCancelled):
            raise
        except (ValueError, RuntimeError, OSError):
            if cancelled and cancelled():
                raise assistant_media.AssistantMediaCancelled("speech evidence cancelled")
            limitations.append("Automatic donor discovery was unavailable; explicitly supplied sources remain pinned.")
        sources = [context.selected, *donors]
        for source in sources:
            if not source.has_audio:
                limitations.append("The selected source has no soundtrack for word localization.")
                continue
            try:
                budget.check()
                with self.factory() as session:
                    artifact = session.get(AssistantArtifact, source.asset_id, populate_existing=True)
                    if artifact is None or artifact.size_bytes > MAX_SPEECH_SOURCE_BYTES:
                        limitations.append("A source exceeded the bounded speech-analysis byte limit and was not transcribed.")
                        continue
                    catalog = self.service.catalog(session)
                    catalog.resolve_editable_reference(target_version_id=context.base_version_id, asset_id=source.asset_id, version_id=source.version_id)
                    path = verify_artifact(artifact, storage=self.service.storage)
                    audio = assistant_audio.probe_trusted_audio(path, timeout_sec=min(30, budget.remaining()), cancelled=budget.check)
                    start = context.selected_passage.start_sec if source.asset_id == context.selected.asset_id and context.selected_passage else 0.0
                    end = min(source.duration_sec, audio.duration_sec, start + 60,
                        context.selected_passage.end_sec if source.asset_id == context.selected.asset_id and context.selected_passage else source.duration_sec)
                    if end <= start:
                        limitations.append("A requested passage lies beyond the available soundtrack; no word timing was inferred.")
                        continue
                    if audio.duration_sec < source.duration_sec - .000001:
                        limitations.append(f"Source {source.asset_id} soundtrack ends at {audio.duration_sec:g} seconds before picture end; recognition uses actual soundtrack coverage.")
                    transcript = transcribe_authorized_artifact(session, catalog=catalog,
                        storage=self.service.storage, target_version_id=context.base_version_id,
                        asset_id=source.asset_id, version_id=source.version_id, asr=self.asr,
                        start_sec=start, end_sec=end, timeout_sec=budget.remaining(), cancelled=budget.check)
                    budget.check()
                    if (abs(transcript.source_start_sec - start) > .000001 or abs(transcript.source_end_sec - end) > .000001
                            or abs(transcript.source_duration_sec - audio.duration_sec) > .1):
                        raise assistant_media.AssistantMediaError("speech evidence changed requested source coverage")
                    item = self._evidence(source, transcript, remaining_words=800 - sum(len(value.words) for value in evidence))
                    evidence.append(item)
                    if not item.coverage_complete:
                        limitations.append(f"Source {source.asset_id} was recognized only from {start:g} to {end:g} seconds; other speech was not analyzed.")
            except (AssistantAccessDenied, LostLease):
                raise
            except (ValueError, RuntimeError, OSError):
                # Cancellation/deadline must propagate; ordinary sidecar/input
                # failures leave an explicit limitation without paid fallback.
                if cancelled and cancelled():
                    raise assistant_media.AssistantMediaCancelled("speech evidence cancelled")
                limitations.append("Offline speech recognition was unavailable for one source; no word timing was inferred for it.")
                if self.monotonic() >= budget.deadline:
                    limitations.append("Reached the 90-second speech-analysis budget; remaining sources were not transcribed.")
                    break
        original_ids = {source.asset_id for source in context.sources}
        admitted_donors = [source for source in donors if source.asset_id not in original_ids]
        # Do not consume the 32-source contract just to attach optional evidence.
        admitted_donors = admitted_donors[:max(0, 32 - len(context.sources))]
        available_ids = {context.selected.asset_id, *(source.asset_id for source in context.sources), *(source.asset_id for source in admitted_donors)}
        payload = context.model_dump(mode="json")
        payload.update(sources=[source.model_dump(mode="json") for source in (*context.sources, *admitted_donors)],
            speech_evidence=[item.model_dump(mode="json") for item in evidence if item.asset_id in available_ids],
            evidence_limitations=list(dict.fromkeys(limitations))[:8])
        while True:
            enriched = PlannerContext.model_validate(payload)
            try:
                context_fingerprint(enriched)  # Same aggregate 64 KiB broker cap.
                return enriched
            except ValueError:
                largest = max(payload["speech_evidence"], key=lambda item: len(item["words"]), default=None)
                if largest and largest["words"]:
                    largest["words"].pop()
                    largest["words_truncated"] = True
                    continue
                # Baseline dialogue may already fill the envelope. Preserve the
                # pinned request and disclose the omitted optional evidence.
                payload["speech_evidence"] = []
                payload["sources"] = [source.model_dump(mode="json") for source in context.sources]
                payload["evidence_limitations"] = ["Speech evidence exceeded the bounded planning context and was omitted."]
                fallback = PlannerContext.model_validate(payload)
                try:
                    context_fingerprint(fallback)
                    return fallback
                except ValueError:
                    return context
