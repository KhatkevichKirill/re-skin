"""Strict bounded proposals and evidence for backend-controlled media reasoning.

These types grant no access. Tools resolve every identifier through the authorized
catalog again; images are data, never filesystem references or model tools.
"""
from __future__ import annotations

import base64
import hashlib
import json
from fractions import Fraction
from typing import Annotated, Literal, Union

from pydantic import Field, StrictBool, StringConstraints, model_validator

from .assistant_contracts import ContractModel, EditPlan, EntityId, Seconds, Duration, ExtendEndingOperation, ending_bounds
from .assistant_planner import AgentReply, CaptionLinesReply, InternalReadDependency, PlannerContext, PlannerReply, PlannerSource, PlannerSpeechEvidence, PlannerSpeechWord, ProposeReply, _reply

Digest = Annotated[str, StringConstraints(strict=True, pattern=r'^[0-9a-f]{64}$')]
MAX_AGENT_INPUT_BYTES = 3 * 1024 * 1024
MAX_FRAME_BYTES = 512 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_CONTEXT_FRAMES = 32
MAX_AGENT_TEXT_BYTES = 64 * 1024
MAX_FRAME_DIMENSION = 768


def plan_fingerprint(plan: EditPlan) -> str:
    return hashlib.sha256(json.dumps(plan.model_dump(mode='json'), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class FrameEvidence(ContractModel):
    frame_id: EntityId
    asset_id: EntityId | None = None
    version_id: EntityId | None = None
    candidate_id: EntityId | None = None
    timestamp_sec: Seconds
    width: Annotated[int, Field(strict=True, ge=1, le=MAX_FRAME_DIMENSION)]
    height: Annotated[int, Field(strict=True, ge=1, le=MAX_FRAME_DIMENSION)]
    source_sha256: Digest
    sha256: Digest
    size_bytes: Annotated[int, Field(strict=True, ge=1, le=MAX_FRAME_BYTES)]
    mime_type: Literal['image/jpeg'] = 'image/jpeg'
    data_base64: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=699052)]

    @model_validator(mode='after')
    def bounded_image(self):
        if (self.asset_id is None) != (self.version_id is None):
            raise ValueError('frame requires both source identifiers')
        if (self.asset_id is None) == (self.candidate_id is None):
            raise ValueError('frame requires exactly one source or candidate')
        try:
            raw = base64.b64decode(self.data_base64, validate=True)
        except (ValueError, TypeError):
            raise ValueError('invalid frame encoding') from None
        if (base64.b64encode(raw).decode('ascii') != self.data_base64 or len(raw) != self.size_bytes
                or hashlib.sha256(raw).hexdigest() != self.sha256 or not raw.startswith(b'\xff\xd8\xff')):
            raise ValueError('frame content does not match its binding')
        # The trusted producer validates pixels with isolated FFmpeg. The
        # credential-bearing transport validates only bounded JPEG structure.
        from .assistant_images import read_jpeg_dimensions
        try:
            dimensions = read_jpeg_dimensions(raw)
        except ValueError:
            raise ValueError('invalid bounded JPEG envelope') from None
        if dimensions != (self.width, self.height) or max(dimensions) > MAX_FRAME_DIMENSION:
            raise ValueError('frame dimensions exceed limit')
        return self


class CandidateEvidence(ContractModel):
    candidate_id: EntityId
    base_version_id: EntityId
    plan: EditPlan
    sha256: Digest
    plan_sha256: Digest
    size_bytes: Annotated[int, Field(strict=True, gt=0)]
    duration_sec: Duration
    has_audio: StrictBool
    subtitle_status: Literal['unknown', 'editable_burned', 'absent']
    mixed_audio_changed: StrictBool
    review_required: Literal[True] = True

    @model_validator(mode='after')
    def pinned_plan(self):
        if self.base_version_id != self.plan.base_version_id or self.plan_sha256 != plan_fingerprint(self.plan):
            raise ValueError('candidate must bind its exact plan')
        return self


class CandidateSpeechEvidence(ContractModel):
    candidate_id: EntityId
    source_sha256: Digest
    source_start_sec: Seconds
    source_end_sec: Duration
    source_duration_sec: Duration
    language: Annotated[str, StringConstraints(strict=True, pattern=r'^[a-z]{2,3}$')]
    language_probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    words: Annotated[tuple[PlannerSpeechWord, ...], Field(max_length=400)] = ()
    coverage_complete: StrictBool
    words_truncated: StrictBool = False
    timing_precision: Literal['estimated_word'] = 'estimated_word'
    pronunciation_verified: Literal[False] = False
    speaker_verified: Literal[False] = False
    review_required: Literal[True] = True

    @model_validator(mode='after')
    def coverage(self):
        if not 0 <= self.source_start_sec < self.source_end_sec <= self.source_duration_sec or self.source_end_sec - self.source_start_sec > 60:
            raise ValueError('invalid candidate speech window')
        if self.coverage_complete != (self.source_start_sec == 0 and self.source_end_sec == self.source_duration_sec):
            raise ValueError('candidate speech must disclose omitted coverage')
        previous = self.source_start_sec
        for word in self.words:
            if word.start_sec < previous or word.end_sec > self.source_end_sec:
                raise ValueError('candidate speech outside covered window')
            previous = word.start_sec
        return self


class InternalProjectRow(ContractModel):
    project_id: EntityId
    project_type: Literal['face_swap', 'subtitle_removal', 'cover_change', 'localisation']
    status: Literal['created', 'analyzing', 'ready', 'failed']


class InternalRunRow(ContractModel):
    run_id: EntityId
    project_id: EntityId
    status: Literal['created', 'preprocessing', 'preprocess_review', 'queued', 'processing', 'stitching', 'delivering', 'done', 'failed', 'incomplete']
    language: Literal['en', 'es', 'pt', 'ja', 'de', 'it', 'pl', 'fr'] | None = None


class InternalVersionRow(ContractModel):
    version_id: EntityId
    run_id: EntityId
    project_id: EntityId
    sequence: Annotated[int, Field(strict=True, ge=1)]
    parent_version_id: EntityId | None = None


class InternalAssetRow(ContractModel):
    asset_id: EntityId
    version_id: EntityId
    run_id: EntityId
    project_id: EntityId
    media_kind: Literal['video', 'audio', 'image']
    sha256: Digest
    size_bytes: Annotated[int, Field(strict=True, gt=0)]
    duration_sec: Duration | None = None
    has_audio: StrictBool | None = None


class _InternalReadPage(ContractModel):
    owner: InternalReadDependency
    next_after_id: EntityId | None = None

    @model_validator(mode='after')
    def bounded_ownership(self):
        expected = {'list_projects': 'project', 'list_runs': 'project', 'list_versions': 'run', 'list_assets': 'version'}[self.kind]
        if self.owner.kind != expected:
            raise ValueError('metadata page has the wrong owner scope')
        for row in self.records:
            if self.kind != 'list_projects' and any(getattr(row, field) != getattr(self.owner, field)
                    for field in ('project_id', 'run_id', 'version_id') if getattr(self.owner, field) is not None):
                raise ValueError('metadata record changed its owner scope')
        identifiers = [getattr(row, {'list_projects': 'project_id', 'list_runs': 'run_id',
            'list_versions': 'version_id', 'list_assets': 'asset_id'}[self.kind]) for row in self.records]
        if identifiers != sorted(set(identifiers)):
            raise ValueError('metadata records require distinct ordered identifiers')
        if len(json.dumps(self.model_dump(mode='json'), ensure_ascii=False, allow_nan=False).encode()) > 8192:
            raise ValueError('metadata page exceeds byte budget')
        return self


class ProjectsReadPage(_InternalReadPage):
    kind: Literal['list_projects']
    records: Annotated[tuple[InternalProjectRow, ...], Field(max_length=20)]


class RunsReadPage(_InternalReadPage):
    kind: Literal['list_runs']
    records: Annotated[tuple[InternalRunRow, ...], Field(max_length=20)]


class VersionsReadPage(_InternalReadPage):
    kind: Literal['list_versions']
    records: Annotated[tuple[InternalVersionRow, ...], Field(max_length=20)]


class AssetsReadPage(_InternalReadPage):
    kind: Literal['list_assets']
    records: Annotated[tuple[InternalAssetRow, ...], Field(max_length=20)]


InternalReadPage = Annotated[Union[ProjectsReadPage, RunsReadPage, VersionsReadPage, AssetsReadPage], Field(discriminator='kind')]


def internal_read_dependencies(page):
    """Derive provenance from whitelisted ownership fields, including empty pages."""
    result = [page.owner]
    kind = {'list_projects': 'project', 'list_runs': 'run', 'list_versions': 'version', 'list_assets': 'asset'}[page.kind]
    for row in page.records:
        values = {key: getattr(row, key) for key in ('project_id', 'run_id', 'version_id', 'asset_id') if hasattr(row, key)}
        dependency = InternalReadDependency(kind=kind, **values)
        if dependency not in result:
            result.append(dependency)
    return tuple(result)


# --- AICP-345: read-only QA tools (app/qa_tools.py) behind the boundary -----

QA_KINDS = ('detect_captions', 'check_alignment', 'compare_frames', 'script_captions')
MAX_QA_CALLS = 4
MAX_QA_OBSERVATION_BYTES = 6 * 1024
MAX_SCRIPT_OBSERVATION_BYTES = 12 * 1024
MAX_QA_HISTORY_BYTES = 24 * 1024
QAScore = Annotated[float, Field(strict=True, ge=0, le=255, allow_inf_nan=False)]
Fraction01 = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
FrameOffset = Annotated[int, Field(strict=True, ge=-60, le=60)]
Drift = Annotated[float, Field(strict=True, ge=-200, le=200, allow_inf_nan=False)]
QALabel = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=24, pattern=r'^[A-Za-z0-9 _.:-]+$')]


class QAMediaRef(ContractModel):
    """One exact authorized input: a pinned source binding or a prior candidate."""
    asset_id: EntityId | None = None
    version_id: EntityId | None = None
    candidate_id: EntityId | None = None
    sha256: Digest

    @model_validator(mode='after')
    def exact(self):
        if (self.asset_id is None) != (self.version_id is None) or (self.asset_id is None) == (self.candidate_id is None):
            raise ValueError('QA input requires one exact source or candidate')
        return self


class QACaptionSpan(ContractModel):
    start_sec: Seconds
    end_sec: Seconds
    frames: Count
    max_score: Count


class QACalibration(ContractModel):
    source: QAMediaRef
    start_sec: Seconds | None = None
    duration_sec: Duration | None = None
    frames: Count
    hits: Count
    hit_ratio: Fraction01
    calibrated: StrictBool
    geometry: dict[Literal['n', 'x0_p5', 'x1_p95', 'y0_p5', 'y0_p50', 'y1_p50', 'y1_p95'],
                   Annotated[float, Field(strict=True, ge=0, le=1_000_000, allow_inf_nan=False)]] | None = None


class CaptionQAEvidence(ContractModel):
    tool: Literal['detect_captions']
    target: QAMediaRef
    start_sec: Seconds | None = None
    duration_sec: Duration | None = None
    config_sha256: Digest
    project_config: StrictBool
    fps: Annotated[float, Field(strict=True, gt=0, le=30, allow_inf_nan=False)]
    threshold: Count
    frames: Count
    hits: Count
    hit_ratio: Fraction01
    verdict: Literal['clean', 'captions_found', 'clean_uncalibrated']
    spans: Annotated[tuple[QACaptionSpan, ...], Field(max_length=24)] = ()
    spans_total: Count
    scores_sha256: Digest
    calibration: QACalibration
    review_required: Literal[True] = True


class QAAlignmentPoint(ContractModel):
    t_sec: Seconds
    offset_frames: FrameOffset
    err: QAScore


class QAAlignmentWindow(ContractModel):
    start_sec: Seconds
    points: Count
    median_offset: Annotated[float, Field(strict=True, ge=-60, le=60, allow_inf_nan=False)]


class QAMatchSample(ContractModel):
    result_t_sec: Seconds
    source_t_sec: Seconds
    drift_sec: Drift
    err: QAScore


class AlignmentQAEvidence(ContractModel):
    tool: Literal['check_alignment']
    mode: Literal['timeline', 'window']
    source: QAMediaRef
    result: QAMediaRef
    # timeline
    source_frames: Count | None = None
    result_frames: Count | None = None
    points: Count | None = None
    nonzero_points: Count | None = None
    outliers: Annotated[tuple[QAAlignmentPoint, ...], Field(max_length=24)] = ()
    outliers_total: Count | None = None
    at_search_limit: Count | None = None
    windows: Annotated[tuple[QAAlignmentWindow, ...], Field(max_length=16)] = ()
    max_abs_window_median: Annotated[float, Field(strict=True, ge=0, le=60, allow_inf_nan=False)] | None = None
    length_ok: StrictBool | None = None
    in_sync: StrictBool | None = None
    tolerance_frames: Count | None = None
    # window (one result moment search inside a source window)
    source_start_sec: Seconds | None = None
    window_duration_sec: Duration | None = None
    result_duration_sec: Seconds | None = None
    samples: Annotated[tuple[QAMatchSample, ...], Field(max_length=40)] = ()
    median_drift_sec: Drift | None = None
    max_abs_drift_sec: Drift | None = None
    detail_sha256: Digest
    review_required: Literal[True] = True

    @model_validator(mode='after')
    def mode_fields(self):
        timeline = (self.source_frames, self.result_frames, self.points, self.outliers_total,
                    self.length_ok, self.in_sync, self.tolerance_frames)
        window = (self.source_start_sec, self.window_duration_sec, self.result_duration_sec)
        if self.mode == 'timeline':
            if any(v is None for v in timeline) or any(v is not None for v in window) or self.samples:
                raise ValueError('timeline alignment evidence is incomplete')
        elif any(v is None for v in window) or any(v is not None for v in timeline) or self.outliers or self.windows:
            raise ValueError('window alignment evidence is incomplete')
        return self


class QACompareSample(ContractModel):
    window: QALabel
    source_t_sec: Seconds
    result_t_sec: Seconds
    diff: QAScore | None = None


class CompareQAEvidence(ContractModel):
    tool: Literal['compare_frames']
    source: QAMediaRef
    result: QAMediaRef
    time_map: Literal['offset', 'scale']
    samples: Annotated[tuple[QACompareSample, ...], Field(min_length=1, max_length=48)]
    review_required: Literal[True] = True


class QAScriptPiece(ContractModel):
    kind: Literal['segment', 'library']
    segment_def_id: EntityId
    run_segment_id: EntityId
    run_id: EntityId
    source_start_sec: Seconds
    source_end_sec: Duration
    clip_duration_sec: Duration
    offset_sec: Seconds
    clip_sha256: Digest
    prompt_sha256: Digest
    script_lines: Count


class QAScriptLine(ContractModel):
    start: Seconds
    end: Duration
    text: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=240)]


class QACaptionTarget(ContractModel):
    caption_id: EntityId
    status: Literal['queued', 'transcribing', 'burning', 'delivering', 'done', 'failed']
    source_type: Literal['run', 'tail_edit']
    language: Annotated[str, StringConstraints(strict=True, max_length=8)] | None = None
    lines_count: Count
    lines_sha256: Digest | None = None
    apply_allowed: StrictBool


class ScriptCaptionsEvidence(ContractModel):
    """Lines from the stored segment prompts, on the run result's timeline."""
    tool: Literal['script_captions']
    lineage: Literal['run_segments', 'library_donor_attestation_required']
    final_sha256: Digest
    final_duration_sec: Duration
    pieces_duration_sec: Duration
    pieces: Annotated[tuple[QAScriptPiece, ...], Field(min_length=1, max_length=12)]
    lines: Annotated[tuple[QAScriptLine, ...], Field(min_length=1, max_length=60)]
    lines_sha256: Digest
    captions: Annotated[tuple[QACaptionTarget, ...], Field(max_length=8)] = ()
    captions_total: Count
    warnings: Annotated[tuple[Literal['library_clip_missing'], ...], Field(max_length=1)] = ()
    script_derived: Literal[True] = True
    review_required: Literal[True] = True

    @model_validator(mode='after')
    def bound(self):
        from .assistant_qa import digest
        if digest([line.model_dump(mode='json') for line in self.lines]) != self.lines_sha256:
            raise ValueError('script caption lines do not match their digest')
        if len({c.caption_id for c in self.captions}) != len(self.captions) or self.captions_total < len(self.captions):
            raise ValueError('invalid caption target list')
        return self


QAEvidence = Annotated[Union[CaptionQAEvidence, AlignmentQAEvidence, CompareQAEvidence, ScriptCaptionsEvidence], Field(discriminator='tool')]


def script_evidence(step, lines_sha256):
    """The latest script_captions evidence of this turn with exactly these lines."""
    for observation in reversed(step.observations):
        if observation.qa is not None and observation.qa.tool == 'script_captions' \
                and observation.qa.lines_sha256 == lines_sha256:
            return observation.qa
    return None


def qa_refs(evidence):
    if evidence.tool == 'script_captions':
        return ()
    if evidence.tool == 'detect_captions':
        return (evidence.target, evidence.calibration.source)
    return (evidence.source, evidence.result)


class AgentObservation(ContractModel):
    kind: Literal['frames', 'speech', 'sources', 'candidate', 'failure', 'internal_read', 'qa']
    action_index: Annotated[int, Field(strict=True, ge=0, le=11)]
    frames: Annotated[tuple[FrameEvidence, ...], Field(max_length=8)] = ()
    speech: Annotated[tuple[PlannerSpeechEvidence, ...], Field(max_length=3)] = ()
    candidate_speech: Annotated[tuple[CandidateSpeechEvidence, ...], Field(max_length=1)] = ()
    sources: Annotated[tuple[PlannerSource, ...], Field(max_length=32)] = ()
    next_after_asset_id: EntityId | None = Field(default=None, exclude_if=lambda value: value is None)
    limitations: Annotated[tuple[Literal['original_source_unavailable', 'source_search_partial', 'source_context_limit'], ...], Field(max_length=3)] = Field(
        default=(), exclude_if=lambda value: not value)
    candidate: CandidateEvidence | None = None
    error: Literal['unavailable', 'invalid_action', 'source_frames_required', 'ending_evidence_required', 'budget_exceeded', 'tool_failed'] | None = None
    internal_read: InternalReadPage | None = Field(default=None, exclude_if=lambda value: value is None)
    qa: QAEvidence | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode='after')
    def exact_payload(self):
        if self.kind != 'sources' and (self.next_after_asset_id is not None or self.limitations):
            raise ValueError('source pagination and limitations require a source observation')
        payload = {'frames': bool(self.frames), 'speech': bool(self.speech or self.candidate_speech), 'sources': bool(self.sources),
                   'candidate': self.candidate is not None, 'failure': self.error is not None,
                   'internal_read': self.internal_read is not None, 'qa': self.qa is not None}
        if any(present for key, present in payload.items() if key != self.kind):
            raise ValueError('observation payload disagrees with kind')
        if self.kind in ('frames', 'speech', 'candidate', 'failure', 'internal_read', 'qa') and not payload[self.kind]:
            raise ValueError('observation requires evidence')
        limit = MAX_SCRIPT_OBSERVATION_BYTES if self.qa is not None and self.qa.tool == 'script_captions' else MAX_QA_OBSERVATION_BYTES
        if self.qa is not None and len(json.dumps(self.qa.model_dump(mode='json'), ensure_ascii=False).encode()) > limit:
            raise ValueError('QA observation exceeds its byte budget')
        return self


class AgentStepContext(ContractModel):
    context: PlannerContext
    step_index: Annotated[int, Field(strict=True, ge=0, le=11)]
    observations: Annotated[tuple[AgentObservation, ...], Field(max_length=12)] = ()

    @model_validator(mode='after')
    def pinned_evidence(self):
        sources = {s.asset_id: s for s in (self.context.selected, *self.context.sources)}
        candidates = {}
        frame_ids = set()
        image_bytes = 0
        previous = -1
        for obs in self.observations:
            if not previous < obs.action_index < self.step_index:
                raise ValueError('observations must be prior ordered actions')
            previous = obs.action_index
            if obs.internal_read is not None and any(dependency not in self.context.internal_read_dependencies
                    for dependency in internal_read_dependencies(obs.internal_read)):
                raise ValueError('metadata observation is outside authorized provenance')
            if obs.qa is not None:
                for ref in qa_refs(obs.qa):
                    if ref.candidate_id is not None:
                        candidate = candidates.get(ref.candidate_id)
                        if candidate is None or candidate.sha256 != ref.sha256:
                            raise ValueError('QA evidence does not match a prior candidate')
                    else:
                        source = sources.get(ref.asset_id)
                        if source is None or source.version_id != ref.version_id or source.media_kind != 'video':
                            raise ValueError('QA evidence does not match authorized video')
            if obs.candidate is not None:
                candidate = obs.candidate
                _validate_plan(candidate.plan, self.context)
                if candidate.candidate_id in candidates:
                    raise ValueError('duplicate candidate')
                candidates[candidate.candidate_id] = candidate
            for source in obs.sources:
                if sources.get(source.asset_id) != source:
                    raise ValueError('discovered sources must be in authorized context')
            for speech in obs.speech:
                source = sources.get(speech.asset_id)
                if source is None or source.version_id != speech.version_id or not source.has_audio:
                    raise ValueError('speech does not match context')
            for speech in obs.candidate_speech:
                candidate = candidates.get(speech.candidate_id)
                if (candidate is None or candidate.sha256 != speech.source_sha256 or not candidate.has_audio
                        or abs(candidate.duration_sec - speech.source_duration_sec) > 0.1):
                    raise ValueError('speech does not match candidate')
            for frame in obs.frames:
                if frame.frame_id in frame_ids:
                    raise ValueError('duplicate frame identifier')
                frame_ids.add(frame.frame_id)
                image_bytes += frame.size_bytes
                if frame.candidate_id is not None:
                    candidate = candidates.get(frame.candidate_id)
                    if candidate is None or candidate.sha256 != frame.source_sha256 or frame.timestamp_sec >= candidate.duration_sec:
                        raise ValueError('frame does not match prior candidate')
                else:
                    source = sources.get(frame.asset_id)
                    if (source is None or source.version_id != frame.version_id or source.media_kind != 'video'
                            or frame.timestamp_sec >= source.duration_sec):
                        raise ValueError('frame does not match authorized video')
        if len(frame_ids) > MAX_CONTEXT_FRAMES or image_bytes > MAX_IMAGE_BYTES:
            raise ValueError('visual evidence exceeds aggregate budget')
        metadata = self.model_dump(mode='json')
        if sum(len(json.dumps(obs.internal_read.model_dump(mode='json'), ensure_ascii=False).encode())
                for obs in self.observations if obs.internal_read is not None) > 24 * 1024:
            raise ValueError('metadata history exceeds aggregate byte budget')
        if sum(len(json.dumps(obs.qa.model_dump(mode='json'), ensure_ascii=False).encode())
                for obs in self.observations if obs.qa is not None) > MAX_QA_HISTORY_BYTES:
            raise ValueError('QA history exceeds aggregate byte budget')
        for obs in metadata['observations']:
            for frame in obs['frames']:
                frame.pop('data_base64')
        if len(json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode()) > MAX_AGENT_TEXT_BYTES:
            raise ValueError('agent text evidence exceeds aggregate budget')
        return self


class _Action(ContractModel):
    run_id: EntityId
    base_version_id: EntityId


class InspectFramesAction(_Action):
    kind: Literal['inspect_frames']
    asset_id: EntityId | None = None
    version_id: EntityId | None = None
    candidate_id: EntityId | None = None
    timestamps_sec: Annotated[tuple[Seconds, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode='after')
    def exact_source(self):
        if (self.asset_id is None) != (self.version_id is None) or (self.asset_id is None) == (self.candidate_id is None):
            raise ValueError('inspect requires one exact source or candidate')
        if tuple(sorted(set(self.timestamps_sec))) != self.timestamps_sec:
            raise ValueError('frame times must be ordered and distinct')
        return self


class TranscribeWindowAction(_Action):
    kind: Literal['transcribe_window']
    asset_id: EntityId | None = None
    version_id: EntityId | None = None
    candidate_id: EntityId | None = None
    start_sec: Seconds
    end_sec: Duration

    @model_validator(mode='after')
    def window(self):
        if (self.asset_id is None) != (self.version_id is None) or (self.asset_id is None) == (self.candidate_id is None):
            raise ValueError('transcribe requires one exact source or candidate')
        if not 0 < self.end_sec - self.start_sec <= 60:
            raise ValueError('speech window must be within 60 seconds')
        return self


class SearchSourcesAction(_Action):
    kind: Literal['search_sources']
    after_asset_id: EntityId | None = Field(default=None, exclude_if=lambda value: value is None)
    query: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=200)]


class ListProjectsAction(_Action):
    kind: Literal['list_projects']
    after_project_id: EntityId | None = None


class ListRunsAction(_Action):
    kind: Literal['list_runs']
    project_id: EntityId
    after_run_id: EntityId | None = None


class ListVersionsAction(_Action):
    kind: Literal['list_versions']
    source_run_id: EntityId
    after_version_id: EntityId | None = None


class ListAssetsAction(_Action):
    kind: Literal['list_assets']
    source_version_id: EntityId
    after_asset_id: EntityId | None = None


INTERNAL_READ_KINDS = ('list_projects', 'list_runs', 'list_versions', 'list_assets')


def internal_read_cursor(action):
    field = dict(zip(INTERNAL_READ_KINDS, ('after_project_id', 'after_run_id', 'after_version_id', 'after_asset_id')))[action.kind]
    return getattr(action, field)


def internal_read_matches(action, page):
    if action.kind != page.kind:
        return False
    return (action.kind == 'list_projects' or
        action.kind == 'list_runs' and page.owner.kind == 'project' and page.owner.project_id == action.project_id or
        action.kind == 'list_versions' and page.owner.kind == 'run' and page.owner.run_id == action.source_run_id or
        action.kind == 'list_assets' and page.owner.kind == 'version' and page.owner.version_id == action.source_version_id)


class _QAInput(ContractModel):
    """An exact authorized QA input; never a path."""
    asset_id: EntityId | None = None
    version_id: EntityId | None = None
    candidate_id: EntityId | None = None

    @model_validator(mode='after')
    def exact(self):
        if (self.asset_id is None) != (self.version_id is None) or (self.asset_id is None) == (self.candidate_id is None):
            raise ValueError('QA input requires one exact source or candidate')
        return self


class QAWindow(ContractModel):
    source_start_sec: Seconds
    source_end_sec: Duration
    result_start_sec: Seconds | None = None
    result_end_sec: Duration | None = None
    label: QALabel | None = None

    @model_validator(mode='after')
    def ordered(self):
        if not self.source_start_sec < self.source_end_sec:
            raise ValueError('window source range must be ordered')
        if (self.result_start_sec is not None and self.result_end_sec is not None
                and not self.result_start_sec < self.result_end_sec):
            raise ValueError('window result range must be ordered')
        return self


class DetectCaptionsAction(_Action):
    """Burned-caption scan of *target*, calibrated on the imported original."""
    kind: Literal['detect_captions']
    target: _QAInput
    calibration: _QAInput
    start_sec: Seconds | None = None
    duration_sec: Duration | None = None
    calibration_start_sec: Seconds | None = None
    calibration_duration_sec: Duration | None = None

    @model_validator(mode='after')
    def sources(self):
        if self.calibration.candidate_id is not None:
            raise ValueError('calibrate on an authorized source, not a candidate')
        return self


class CheckAlignmentAction(_Action):
    """Picture alignment of *result* against *source* (timeline or one window)."""
    kind: Literal['check_alignment']
    source: _QAInput
    result: _QAInput
    mode: Literal['timeline', 'window'] = 'timeline'
    source_start_sec: Seconds | None = None
    duration_sec: Duration | None = None

    @model_validator(mode='after')
    def window(self):
        if self.source.candidate_id is not None:
            raise ValueError('alignment source must be an authorized source')
        if self.mode == 'window':
            if self.source_start_sec is None or self.duration_sec is None or self.duration_sec > 30:
                raise ValueError('window alignment requires source_start_sec and duration_sec <= 30')
        elif self.source_start_sec is not None:
            raise ValueError('timeline alignment compares both inputs from 0')
        return self


class CompareFramesAction(_Action):
    """Per-pair numeric source/result differences at sampled moments."""
    kind: Literal['compare_frames']
    source: _QAInput
    result: _QAInput
    windows: Annotated[tuple[QAWindow, ...], Field(min_length=1, max_length=8)]
    fractions: Annotated[tuple[Fraction01, ...], Field(min_length=1, max_length=6)] | None = (0.1, 0.5, 0.9)
    every_sec: Annotated[float, Field(strict=True, ge=0.25, le=10, allow_inf_nan=False)] | None = None
    time_map: Literal['offset', 'scale'] = 'offset'

    @model_validator(mode='after')
    def sampling(self):
        if self.source.candidate_id is not None:
            raise ValueError('comparison source must be an authorized source')
        if self.every_sec is None and not self.fractions:
            raise ValueError('compare requires fractions or every_sec')
        return self


class LibraryDonor(ContractModel):
    plan_index: Annotated[int, Field(strict=True, ge=0, le=11)]
    run_segment_id: EntityId


class ScriptCaptionsAction(_Action):
    """Captions from the run's stored segment prompts (read-only)."""
    kind: Literal['script_captions']
    library_donors: Annotated[tuple[LibraryDonor, ...], Field(max_length=4)] = ()

    @model_validator(mode='after')
    def distinct(self):
        if len({d.plan_index for d in self.library_donors}) != len(self.library_donors):
            raise ValueError('one donor per library entry')
        return self


class RenderCandidateAction(_Action):
    kind: Literal['render_candidate']
    plan: EditPlan


class FinalAction(_Action):
    kind: Literal['final']
    reply: AgentReply
    candidate_id: EntityId | None = None


AgentAction = Annotated[Union[InspectFramesAction, TranscribeWindowAction, SearchSourcesAction,
    ListProjectsAction, ListRunsAction, ListVersionsAction, ListAssetsAction,
    DetectCaptionsAction, CheckAlignmentAction, CompareFramesAction, ScriptCaptionsAction,
    RenderCandidateAction, FinalAction], Field(discriminator='kind')]


class AgentActionEnvelope(ContractModel):
    action: AgentAction


def _validate_plan(plan, context):
    _reply(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
        'structured_output': {'reply': {'kind': 'propose', 'run_id': context.run_id,
        'base_version_id': context.base_version_id, 'explanation': 'Candidate plan',
        'plan': plan.model_dump(mode='json')}}}).encode(), context, allow_ending=True)


def _ending_frame_clock(operation, context):
    """Use the same rational frame boundaries as the renderer, not probe decimals."""
    start, end = ending_bounds(operation,context.selected.duration_sec,context.selected.fps)
    rate = Fraction(context.selected.fps or '30')
    base_end = Fraction(str(context.selected.duration_sec))
    if context.selected.fps is not None:
        base_end = round(base_end*rate)/rate
    return start,end,base_end,rate


def ending_source_evidence(plan, step):
    """Source pixels and complete donor speech must precede ending rendering."""
    operation = plan.operations[0]
    if not isinstance(operation,ExtendEndingOperation):
        return True
    context = step.context
    start_clock, end_clock, base_end, rate = _ending_frame_clock(operation,context)
    start, end = float(start_clock),float(end_clock)
    margin = max(Fraction(1,10),2/rate)
    sampled = [frame.timestamp_sec for observation in step.observations for frame in observation.frames
        if frame.asset_id == context.selected.asset_id and frame.version_id == context.selected.version_id]
    if not any(float(max(0,start_clock-margin)) <= timestamp < start for timestamp in sampled):
        return False
    if not any(start <= timestamp < float(min(base_end,start_clock+margin)) for timestamp in sampled):
        return False
    frozen_at = float(round(Fraction(str(operation.visual.timestamp_sec))*rate)/rate) if operation.visual.kind == 'freeze_selected' else None
    if frozen_at is not None and not any(abs(timestamp-frozen_at) <= 1e-6 for timestamp in sampled):
        return False
    donor_sample = round(Fraction(str(operation.donor_start_sec))*48000)
    donor_start = donor_sample/48000
    donor_end = (donor_sample+round(end_clock*48000)-round(start_clock*48000))/48000
    covered = sorted((speech.source_start_sec,speech.source_end_sec) for observation in step.observations
        for speech in observation.speech if speech.asset_id == operation.source.asset_id
        and speech.version_id == operation.source.version_id and not speech.words_truncated)
    cursor = donor_start
    for left,right in covered:
        if left <= cursor and right > cursor:
            cursor = right
        if cursor >= donor_end:
            return True
    return False


def candidate_visual_windows(plan: EditPlan, context: PlannerContext, duration_sec: float):
    """Output-timeline windows requiring sampled pixels, never exhaustive QA."""
    windows = []
    seam_margin = max(0.1, 2 / float(Fraction(context.selected.fps or '30')))
    for operation in plan.operations:
        if isinstance(operation,ExtendEndingOperation):
            start, end, base_end, rate = _ending_frame_clock(operation,context)
            margin = max(Fraction(1,10),2/rate)
            # Frame inspection reports n/fps. Six-decimal probe durations can
            # round above the sole added frame, making a valid one-frame
            # extension impossible to inspect if used as the window boundary.
            ending_windows = [(max(0,start-margin),min(end,start+margin)),
                (start,base_end),(base_end,end),(max(start,end-Fraction(3,2)/rate),end)]
            if operation.fade_out_sec:
                fade = round(Fraction(str(operation.fade_out_sec))*rate)/rate
                ending_windows.append((end-fade,end-Fraction(3,2)/rate))
            windows.extend((float(left),float(right)) for left,right in ending_windows)
        elif operation.kind in ('image_fade', 'replace_footage'):
            windows.append((operation.start_sec, min(duration_sec, operation.end_sec)))
        elif operation.kind == 'cut':
            # Removed input interval collapses to this seam in the output.
            seam = operation.start_sec
            windows.append((max(0.0, seam - seam_margin), min(duration_sec, seam + seam_margin)))
        elif operation.kind == 'trim':
            if operation.start_sec > 0:
                windows.append((0.0, min(duration_sec, seam_margin)))
            if operation.end_sec < context.selected.duration_sec:
                windows.append((max(0.0, duration_sec - seam_margin), duration_sec))
        elif operation.kind == 'subtitle_edit':
            original = {cue.cue_id: cue for cue in context.subtitle_cues}
            for change in operation.changes:
                # A moved cue changes both its old and new passages: absence
                # at the old location needs sampled pixels too.
                old_cue = original.get(change.cue_id)
                if old_cue is not None:
                    windows.append((old_cue.start_sec, min(duration_sec, old_cue.end_sec)))
                if change.action == 'update':
                    windows.append((change.start_sec, min(duration_sec, change.end_sec)))
            if operation.style is not None:
                # Style affects every existing cue even when combined with
                # individual text/timing edits in the same operation.
                windows.extend((cue.start_sec, min(duration_sec, cue.end_sec)) for cue in context.subtitle_cues)
    return tuple(dict.fromkeys(windows))


def validate_agent_action(action: AgentAction, step: AgentStepContext):
    action = AgentActionEnvelope.model_validate({'action': action.model_dump(mode='python')}).action
    context = step.context
    if (action.run_id, action.base_version_id) != (context.run_id, context.base_version_id):
        raise ValueError('action changed pinned scope')
    sources = {s.asset_id: s for s in (context.selected, *context.sources)}
    if isinstance(action, (InspectFramesAction, TranscribeWindowAction)):
        if action.candidate_id is not None:
            candidates = [o.candidate for o in step.observations if o.candidate is not None
                          and o.candidate.candidate_id == action.candidate_id]
            if not candidates:
                raise ValueError('unknown candidate')
            if isinstance(action, InspectFramesAction):
                if any(t >= candidates[0].duration_sec for t in action.timestamps_sec):
                    raise ValueError('invalid candidate frame time')
            elif not candidates[0].has_audio or action.end_sec > candidates[0].duration_sec:
                raise ValueError('invalid candidate speech window')
        else:
            source = sources.get(action.asset_id)
            if source is None or source.version_id != action.version_id:
                raise ValueError('unknown source binding')
            if isinstance(action, InspectFramesAction):
                if source.media_kind != 'video' or any(t >= source.duration_sec for t in action.timestamps_sec):
                    raise ValueError('invalid video frame time')
            elif not source.has_audio or action.end_sec > source.duration_sec:
                raise ValueError('invalid speech source window')
    elif action.kind == 'script_captions':
        pass
    elif action.kind in QA_KINDS:
        inputs = ((action.target, action.start_sec, action.duration_sec),
                  (action.calibration, action.calibration_start_sec, action.calibration_duration_sec)) \
            if action.kind == 'detect_captions' else ((action.source, None, None), (action.result, None, None))
        for item, start, duration in inputs:
            if item.candidate_id is not None:
                found = [o.candidate for o in step.observations if o.candidate is not None
                         and o.candidate.candidate_id == item.candidate_id]
                if not found:
                    raise ValueError('unknown candidate')
                length = found[0].duration_sec
            else:
                source = sources.get(item.asset_id)
                if source is None or source.version_id != item.version_id or source.media_kind != 'video':
                    raise ValueError('unknown source binding')
                length = source.duration_sec
            if (start or 0) >= length or (start or 0) + (duration or 0) > length + 1e-6:
                raise ValueError('QA window outside its input')
        if action.kind == 'check_alignment' and action.mode == 'window':
            source = sources[action.source.asset_id]
            if action.source_start_sec + action.duration_sec > source.duration_sec + 1e-6:
                raise ValueError('QA window outside its input')
        if action.kind == 'compare_frames':
            source = sources[action.source.asset_id]
            if any(w.source_end_sec > source.duration_sec + 1e-6 for w in action.windows):
                raise ValueError('QA window outside its input')
    elif isinstance(action, SearchSourcesAction):
        if action.after_asset_id is not None and not any(
                observation.kind == 'sources' and observation.next_after_asset_id == action.after_asset_id
                for observation in step.observations):
            raise ValueError('source pagination requires a previously supplied catalog cursor')
    elif action.kind in INTERNAL_READ_KINDS:
        cursor = internal_read_cursor(action)
        if cursor is not None and not any(observation.internal_read is not None
                and internal_read_matches(action, observation.internal_read)
                and observation.internal_read.next_after_id == cursor for observation in step.observations):
            raise ValueError('metadata pagination requires its previously supplied scoped cursor')
    elif isinstance(action, RenderCandidateAction):
        _validate_plan(action.plan, context)
    elif isinstance(action, FinalAction) and isinstance(action.reply, CaptionLinesReply):
        reply = action.reply
        if (reply.run_id, reply.base_version_id) != (context.run_id, context.base_version_id):
            raise ValueError('caption proposal changed pinned scope')
        if action.candidate_id is not None:
            raise ValueError('caption proposals bind no candidate')
        script = script_evidence(step, reply.lines_sha256)
        if script is None:
            raise ValueError('caption proposal requires script_captions evidence of this turn')
        if not any(c.caption_id == reply.caption_id and c.apply_allowed for c in script.captions):
            raise ValueError('caption proposal must name an applicable caption of the evidence')
    elif isinstance(action, FinalAction):
        _reply(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
            'structured_output': {'reply': action.reply.model_dump(mode='json')}}).encode(), context, allow_ending=True)
        if isinstance(action.reply, ProposeReply):
            candidates = [o.candidate for o in step.observations if o.candidate is not None
                          and o.candidate.candidate_id == action.candidate_id]
            if not candidates or candidates[0].plan != action.reply.plan:
                raise ValueError('final proposal requires exact rendered candidate')
            if isinstance(action.reply.plan.operations[0],ExtendEndingOperation):
                _, ending_end = ending_bounds(action.reply.plan.operations[0],context.selected.duration_sec,context.selected.fps)
                if (not ending_source_evidence(action.reply.plan,step) or not candidates[0].mixed_audio_changed
                        or not candidates[0].has_audio or abs(candidates[0].duration_sec-ending_end) > 1e-6):
                    raise ValueError('ending proposal requires source evidence and exact extended candidate')
            if not any(f.candidate_id == action.candidate_id for o in step.observations for f in o.frames):
                raise ValueError('final proposal requires candidate frame evidence')
            sampled = [frame.timestamp_sec for observation in step.observations for frame in observation.frames
                       if frame.candidate_id == action.candidate_id]
            for start, end in candidate_visual_windows(action.reply.plan, context, candidates[0].duration_sec):
                if not any(start <= timestamp < end for timestamp in sampled):
                    raise ValueError('final proposal requires candidate frames at each affected visual interval or output seam')
            required = []
            for operation in action.reply.plan.operations:
                if operation.kind == 'full_track_audio':
                    required.append((0.0, candidates[0].duration_sec))
                elif operation.kind == 'audio_splice':
                    required.append((operation.start_sec, operation.end_sec))
                elif isinstance(operation,ExtendEndingOperation):
                    start, _ = ending_bounds(operation,context.selected.duration_sec,context.selected.fps)
                    required.append((float(start),candidates[0].duration_sec))
            speech_evidence = [speech for observation in step.observations for speech in observation.candidate_speech
                if speech.candidate_id == action.candidate_id and not speech.words_truncated]
            covered = sorted((speech.source_start_sec, speech.source_end_sec) for speech in speech_evidence)
            # AAC may genuinely stop up to 50 ms before the video. Only
            # source-duration metadata from this exact candidate can establish
            # that missing tail; uncovered transcription gaps stay forbidden.
            audio_extents = {speech.source_duration_sec for speech in speech_evidence}
            if len(audio_extents) > 1:
                raise ValueError('candidate speech windows disagree on the source audio extent')
            audio_end = next(iter(audio_extents), candidates[0].duration_sec)
            video_end = candidates[0].duration_sec
            proved_short_tail = 0 <= video_end - audio_end <= 0.05 + 1e-9
            for start, end in required:
                if proved_short_tail and audio_end < end <= video_end:
                    end = audio_end
                cursor = start
                for left, right in covered:
                    if left <= cursor and right > cursor:
                        cursor = right
                    if cursor >= end:
                        break
                if cursor < end:
                    raise ValueError('final audio proposal requires complete changed-window candidate speech evidence')
        elif action.candidate_id is not None:
            raise ValueError('candidate binding requires a proposal')
    return action
