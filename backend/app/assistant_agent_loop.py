"""Bounded backend-controlled tool loop. Unknown work is never replayed."""
from __future__ import annotations

import base64
import logging
from datetime import timedelta
from sqlalchemy import or_, select, update
from .assistant_agent_steps import AssistantAgentStep, begin_step, save_action, complete_step
from .assistant_jobs import LostLease
from .assistant_broker_client import BrokerError
from .assistant_turns import AssistantTurn, claim_turn, fail_turn, finalize_cancel, renew_turn


class AgentLoop:
    def __init__(self, supervisor):
        self.s = supervisor

    def cleanup(self):
        s = self.s
        with s.factory() as session:
            ids = session.scalars(select(AssistantAgentStep.id).join(AssistantTurn,
                AssistantTurn.id == AssistantAgentStep.turn_id).where(AssistantAgentStep.cleanup_pending.is_(True),
                or_(AssistantTurn.status.in_(("succeeded", "failed", "cancelled")), AssistantAgentStep.status != 'submitted'),
                or_(AssistantAgentStep.cleanup_next_at.is_(None), AssistantAgentStep.cleanup_next_at <= s.clock()))
                .order_by(AssistantAgentStep.created_at).limit(12)).all()
            cancelling = session.scalars(select(AssistantAgentStep.id).join(AssistantTurn,
                AssistantTurn.id == AssistantAgentStep.turn_id).where(AssistantAgentStep.cleanup_pending.is_(True),
                AssistantTurn.status == 'cancel_requested', AssistantAgentStep.status == 'submitted').limit(4)).all()
        for step_id in cancelling:
            s._safe_cancel(step_id)
        for step_id in ids:
            with s.factory() as session:
                row = session.get(AssistantAgentStep, step_id)
                fingerprint = row.context_sha256
            done = False
            try:
                result = s.broker.poll(step_id)
                if (result.turn_id, result.context_sha256) != (step_id, fingerprint):
                    raise ValueError('broker cleanup binding mismatch')
                if result.state in ('queued','running','cancel_requested'):
                    result = s.broker.cancel(step_id)
                if (result.turn_id, result.context_sha256) == (step_id, fingerprint) and result.state in ('succeeded','failed','cancelled','interrupted'):
                    result = s.broker.acknowledge(step_id)
                    done = ((result.turn_id, result.context_sha256) == (step_id, fingerprint)
                        and result.acknowledged and result.state in ('succeeded','failed','cancelled','interrupted'))
            except BrokerError as error:
                if error.code == 'not_found':
                    done = s._fence_missing(step_id, fingerprint)
            except ValueError:
                pass
            with s.factory() as session:
                session.execute(update(AssistantAgentStep).where(AssistantAgentStep.id == step_id,
                    AssistantAgentStep.context_sha256 == fingerprint, AssistantAgentStep.cleanup_pending.is_(True))
                    .values(cleanup_pending=not done, cleanup_next_at=None if done else s.clock()+timedelta(seconds=5)))
                session.commit()

    def run(self, turn_id):
        from .assistant_agent_contracts import AgentStepContext, validate_agent_action, FinalAction, plan_fingerprint, INTERNAL_READ_KINDS
        from .assistant_agent_contracts import MAX_QA_CALLS, MAX_QA_HISTORY_BYTES, QA_KINDS
        from .assistant_agent_media import AgentToolUnavailable
        from .assistant_supervisor import _PlanningLeaseGuard
        s, step_id = self.s, None
        self.turn_id = turn_id
        with s.factory() as session:
            lease = claim_turn(session, turn_id, owner=s.owner, now=s.clock(), lease_sec=s.lease_sec)
            session.commit()
        if lease is None:
            return
        guard = _PlanningLeaseGuard(s, lease)
        started = s.monotonic()

        def check():
            if s.monotonic() - started > s.agent_deadline_sec:
                raise TimeoutError('agent deadline exceeded')
            if guard():
                raise InterruptedError('agent cancelled')

        def authorize(context):
            check()
            with s.factory() as session:
                # SQL ownership and authority are checked for each inference and tool.
                s.service.load_context(session, guard.lease, now=s.clock())
                s.service.reauthorize_context(session, context)
                guard.lease = renew_turn(session, guard.lease, now=s.clock(), lease_sec=s.lease_sec)
                session.commit()
                guard.renewed = s.monotonic()

        try:
            with s.factory() as session:
                context = s.service.load_context(session, lease, now=s.clock())
                session.commit()
            observations = ()
            with s.media_factory(context=context, cancelled=check) as media:
                for index in range(s.agent_max_steps):
                    authorize(context)
                    step = AgentStepContext(context=context, step_index=index, observations=select_evidence(observations))
                    with s.factory() as session:
                        step_id, fingerprint = begin_step(session, guard.lease, step, now=s.clock())
                        session.commit()
                    result = s.broker.submit(step_id, context, agent_step=step)
                    while True:
                        check()
                        if (result.turn_id, result.context_sha256) != (step_id, fingerprint):
                            raise ValueError('broker changed the pinned agent step')
                        if result.state == 'succeeded':
                            if result.action is None or result.acknowledged:
                                raise ValueError('agent result missing unacknowledged action')
                            action = validate_agent_action(result.action, step)
                            break
                        if result.state in ('failed','cancelled','interrupted'):
                            raise BrokerError(result.error_code or result.state)
                        s.sleep(.5)
                        result = s.broker.poll(step_id)
                    authorize(context)
                    with s.factory() as session:
                        previous = session.scalars(select(AssistantAgentStep).where(
                            AssistantAgentStep.turn_id == turn_id, AssistantAgentStep.action.is_not(None))).all()
                        if any(row.action == action.model_dump(mode='json') and (row.observation or {}).get('kind') != 'failure' for row in previous):
                            raise ValueError('repeated agent action')
                        kinds = [row.action['kind'] for row in previous]
                        limits = {'render_candidate': 3, 'search_sources': 2, 'transcribe_window': 3}
                        if action.kind in INTERNAL_READ_KINDS and sum(kind in INTERNAL_READ_KINDS for kind in kinds) >= 4:
                            raise TimeoutError('agent metadata call budget exceeded')
                        if action.kind in QA_KINDS and sum(kind in QA_KINDS for kind in kinds) >= MAX_QA_CALLS:
                            raise TimeoutError('agent QA call budget exceeded')
                        if action.kind in limits and kinds.count(action.kind) >= limits[action.kind]:
                            raise TimeoutError('agent tool budget exceeded')
                        save_action(session, guard.lease, step_id, action, now=s.clock())
                        session.commit()
                    self.cleanup()
                    authorize(context)
                    if isinstance(action, FinalAction):
                        private = None
                        if action.candidate_id is not None:
                            candidate = next(o.candidate for o in observations if o.candidate is not None
                                and o.candidate.candidate_id == action.candidate_id)
                            retained = media.persist_candidate(action.candidate_id)
                            if (retained.get('sha256') != candidate.sha256
                                    or retained.get('size_bytes') != candidate.size_bytes
                                    or candidate.plan_sha256 != plan_fingerprint(action.reply.plan)):
                                raise ValueError('retained candidate changed its exact evidence binding')
                            private = dict(retained, candidate_id=action.candidate_id,
                                plan_sha256=candidate.plan_sha256, candidate_rendered=True,
                                candidate_frames_inspected=True, operator_review_required=True,
                                inspected_timestamps_sec=[frame.timestamp_sec for observation in step.observations
                                    for frame in observation.frames if frame.candidate_id == action.candidate_id],
                                candidate_speech_inspected=any(speech.candidate_id == action.candidate_id
                                    for observation in step.observations for speech in getattr(observation, 'candidate_speech', ())))
                        authorize(context)
                        with s.factory() as session:
                            complete_step(session, guard.lease, step_id, private_artifact=private, now=s.clock())
                            s.service.save_agent_context(session, guard.lease, context, now=s.clock())
                            s.service.finish(session, guard.lease, action.reply, now=s.clock())
                            session.commit()
                        return
                    try:
                        observation, context = self.tool(media, action, index, context)
                    except AgentToolUnavailable:
                        from .assistant_agent_contracts import AgentObservation
                        observation = AgentObservation(kind='failure', action_index=index, error='unavailable')
                    authorize(context)
                    if observation.qa is not None:
                        import json
                        if sum(len(json.dumps(item.qa.model_dump(mode='json'), ensure_ascii=False).encode())
                                for item in (*observations, observation) if item.qa is not None) > MAX_QA_HISTORY_BYTES:
                            raise TimeoutError('agent QA byte budget exceeded')
                    if observation.internal_read is not None:
                        import json
                        pages = [item.internal_read for item in (*observations, observation) if item.internal_read is not None]
                        if sum(len(json.dumps(page.model_dump(mode='json'), ensure_ascii=False).encode()) for page in pages) > 24 * 1024:
                            raise TimeoutError('agent metadata byte budget exceeded')
                    # Validate aggregate evidence budget before making it durable.
                    if index + 1 < 12:
                        AgentStepContext(context=context, step_index=index+1, observations=select_evidence((*observations, observation)))
                    with s.factory() as session:
                        complete_step(session, guard.lease, step_id, observation=observation, now=s.clock())
                        session.commit()
                    observations = (*observations, observation)
                raise TimeoutError('agent step budget exhausted')
        except Exception as error:
            logging.getLogger(__name__).warning('Assistant agent loop ended: %s', type(error).__name__)
            if step_id is not None:
                s._safe_cancel(step_id)
            with s.factory() as session:
                try:
                    turn = session.get(AssistantTurn, turn_id, populate_existing=True)
                    if turn is not None and turn.status == 'cancel_requested':
                        # The tool context has exited and a broker terminal state
                        # is required before this owner attests cancellation.
                        if step_id is None:
                            finalize_cancel(session, turn_id, lease=guard.lease, now=s.clock())
                        else:
                            result = s.broker.cancel(step_id)
                            if result.turn_id == step_id and result.context_sha256 == fingerprint and result.state in ('succeeded','failed','cancelled','interrupted'):
                                finalize_cancel(session, turn_id, lease=guard.lease, now=s.clock())
                    elif not isinstance(error, LostLease):
                        fail_turn(session, guard.lease, error_code='agent_budget_exhausted' if isinstance(error, TimeoutError)
                            else 'supervisor_stopping' if s.stopping() else 'runtime_' + error.code if isinstance(error, BrokerError) else 'agent_failed', now=s.clock())
                    session.commit()
                except (LostLease, BrokerError):
                    session.rollback()
        finally:
            self.cleanup()

    def tool(self, media, action, index, context):
        from .assistant_agent_contracts import AgentObservation, FrameEvidence, CandidateEvidence, INTERNAL_READ_KINDS, QA_KINDS
        from .assistant_planner import PlannerSource
        if action.kind in INTERNAL_READ_KINDS:
            from .assistant_internal_api import read_internal_metadata
            with self.s.factory() as session:
                page, context = read_internal_metadata(session, service=self.s.service, context=context,
                    action=action, cancelled=media.check if hasattr(media, 'check') else None)
            observation = AgentObservation(kind='internal_read', action_index=index, internal_read=page)
        elif action.kind == 'inspect_frames':
            with self.s.factory() as session:
                rows = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == self.turn_id, AssistantAgentStep.action.is_not(None))).all()
                if sum(len(row.action.get('timestamps_sec', [])) for row in rows) > 32:
                    raise TimeoutError('agent image budget exceeded')
            output = media.inspect_frames(asset_id=action.asset_id, version_id=action.version_id,
                candidate_id=action.candidate_id, timestamps_sec=action.timestamps_sec)
            frames = []
            for item in output['frames']:
                raw = media.read_frame(item['frame_id'])
                item = {key: value for key, value in item.items() if key in FrameEvidence.model_fields}
                item['data_base64'] = base64.b64encode(raw).decode('ascii')
                frames.append(FrameEvidence.model_validate(item))
            observation = AgentObservation(kind='frames', action_index=index, frames=tuple(frames))
        elif action.kind == 'search_sources':
            output = media.search_sources(query=action.query, after_asset_id=action.after_asset_id)
            incoming = tuple(PlannerSource.model_validate(source) for source in output['sources'])
            known = {source.asset_id: source for source in context.sources}
            sources = []
            limitations = list(output.get('limitations', ()))
            if output.get('next_after_asset_id') is not None:
                limitations.append('source_search_partial')
            for source in incoming:
                existing = context.selected if source.asset_id == context.selected.asset_id else known.get(source.asset_id)
                if existing is not None:
                    if existing.model_dump(exclude={'label'}) != source.model_dump(exclude={'label'}):
                        raise ValueError('discovery changed a pinned source')
                    source = existing
                if source.asset_id != context.selected.asset_id:
                    if source.asset_id not in known and len(known) >= 32:
                        limitations.append('source_context_limit')
                        continue
                    known[source.asset_id] = source
                sources.append(source)
            sources = tuple(sources)
            context = context.model_copy(update={'sources': tuple(known.values())})
            observation = AgentObservation(kind='sources', action_index=index, sources=sources,
                next_after_asset_id=output.get('next_after_asset_id'),
                limitations=tuple(dict.fromkeys(limitations)))
        elif action.kind == 'transcribe_window':
            output = media.transcribe_window(asset_id=action.asset_id, version_id=action.version_id,
                candidate_id=action.candidate_id, start_sec=action.start_sec, end_sec=action.end_sec)
            payload = {}
            for field in ('speech','candidate_speech'):
                if field in output:
                    value = output[field]
                    payload[field] = tuple(value) if isinstance(value, (list,tuple)) else (value,)
            observation = AgentObservation(kind='speech', action_index=index, **payload)
        elif action.kind in QA_KINDS:
            from pydantic import TypeAdapter
            from .assistant_agent_contracts import QAEvidence
            if action.kind == 'detect_captions':
                output = media.detect_captions(target=action.target.model_dump(), calibration=action.calibration.model_dump(),
                    start_sec=action.start_sec, duration_sec=action.duration_sec,
                    calibration_start_sec=action.calibration_start_sec,
                    calibration_duration_sec=action.calibration_duration_sec)
            elif action.kind == 'check_alignment':
                output = media.check_alignment(source=action.source.model_dump(), result=action.result.model_dump(),
                    mode=action.mode, source_start_sec=action.source_start_sec, duration_sec=action.duration_sec)
            elif action.kind == 'script_captions':
                output = media.script_captions(library_donors=[donor.model_dump() for donor in action.library_donors])
            else:
                output = media.compare_frames(source=action.source.model_dump(), result=action.result.model_dump(),
                    windows=[window.model_dump() for window in action.windows], fractions=action.fractions,
                    every_sec=action.every_sec, time_map=action.time_map)
            observation = AgentObservation(kind='qa', action_index=index,
                qa=TypeAdapter(QAEvidence).validate_python(output))
        elif action.kind == 'render_candidate':
            with self.s.factory() as session:
                rows = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == self.turn_id,
                    AssistantAgentStep.status == 'completed')).all()
                if not any(frame.get('asset_id') == context.selected.asset_id
                    and frame.get('version_id') == context.selected.version_id
                    for row in rows for frame in (row.observation or {}).get('frames', [])):
                    from .assistant_agent_contracts import AgentObservation
                    return AgentObservation(kind='failure', action_index=index, error='source_frames_required'), context
                from .assistant_agent_contracts import ending_source_evidence
                from .assistant_contracts import ExtendEndingOperation
                from .assistant_agent_steps import restore_step
                current = session.scalar(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == self.turn_id,
                    AssistantAgentStep.step_index == index))
                if (isinstance(action.plan.operations[0],ExtendEndingOperation) and
                        (current is None or not ending_source_evidence(action.plan,restore_step(session,current)))):
                    return AgentObservation(kind='failure',action_index=index,error='ending_evidence_required'), context
            output = media.render_candidate(action.plan)['candidate']
            output = {key: value for key, value in output.items() if key in CandidateEvidence.model_fields}
            output['plan'] = action.plan
            observation = AgentObservation(kind='candidate', action_index=index,
                candidate=CandidateEvidence.model_validate(output))
        else:
            raise ValueError('unsupported agent action')
        return observation, context


def _spread_frames(frames):
    """Prioritize temporal endpoints, then the least represented time region."""
    ordered = sorted(frames, key=lambda frame: (frame.timestamp_sec, frame.frame_id))
    if len(ordered) <= 1:
        return ordered
    chosen = [ordered.pop(0), ordered.pop(-1)]
    while ordered:
        # Stable ascending ties make repeated compaction deterministic.
        best = max(range(len(ordered)), key=lambda index: min(
            abs(ordered[index].timestamp_sec - picked.timestamp_sec) for picked in chosen))
        chosen.append(ordered.pop(best))
    return chosen


def select_evidence(observations):
    """Deliver fresh requested images, then balance historical comparisons.

    SQL keeps every original frame. The last inspection always informs the next
    inference; remaining slots sample the current candidate and source histories.
    """
    from .assistant_agent_contracts import MAX_IMAGE_BYTES, MAX_CONTEXT_FRAMES
    all_frames = [frame for observation in observations for frame in observation.frames]
    if len(all_frames) <= MAX_CONTEXT_FRAMES and sum(frame.size_bytes for frame in all_frames) <= MAX_IMAGE_BYTES:
        return tuple(observations)
    candidates, groups, seen = {}, {}, set()
    latest_candidate = None
    for observation in observations:
        for frame in observation.frames:
            if frame.frame_id in seen:
                continue
            seen.add(frame.frame_id)
            if frame.candidate_id is not None:
                latest_candidate = frame.candidate_id
                candidates.setdefault(frame.candidate_id, []).append(frame)
            else:
                groups.setdefault((frame.asset_id, frame.version_id), []).append(frame)
    selected, used = set(), 0

    def take(frame):
        nonlocal used
        if frame.frame_id in selected:
            return False
        if len(selected) >= MAX_CONTEXT_FRAMES or used + frame.size_bytes > MAX_IMAGE_BYTES:
            return False
        selected.add(frame.frame_id)
        used += frame.size_bytes
        return True

    fresh = next((observation for observation in reversed(observations) if observation.frames), None)
    if fresh is not None:
        for frame in fresh.frames:
            take(frame)
    queues = []
    if latest_candidate is not None:
        queues.append(_spread_frames([frame for frame in candidates[latest_candidate]
            if frame.frame_id not in selected]))
    queues.extend(_spread_frames([frame for frame in frames if frame.frame_id not in selected])
        for frames in groups.values())
    queues.extend(_spread_frames([frame for frame in frames if frame.frame_id not in selected])
        for candidate_id, frames in candidates.items() if candidate_id != latest_candidate)
    # Reserve newly requested evidence first. Balance remaining historical
    # samples so a small follow-up can still compare sources and the candidate.
    while any(queues) and len(selected) < MAX_CONTEXT_FRAMES:
        for queue in queues:
            while queue:
                frame = queue.pop(0)
                if take(frame):
                    break
    chosen = []
    for observation in observations:
        if observation.kind != 'frames':
            chosen.append(observation)
            continue
        frames = tuple(frame for frame in observation.frames if frame.frame_id in selected)
        if frames:
            chosen.append(observation.model_copy(update={'frames': frames}))
    return tuple(chosen)
