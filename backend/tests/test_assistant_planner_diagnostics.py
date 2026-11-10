"""Fixed-code diagnostics use offline outputs/fakes; never native inference."""
import json
import logging

import pytest

from app import assistant_planner as planner
from app.assistant_agent_contracts import AgentStepContext
from app.assistant_broker import _log_planner_failure
from tests.test_assistant_broker import setup_broker, terminal, TURN
from tests.test_assistant_planner import context, fake, RUN, BASE, ASSET, OTHER

SECRET = 'private-input-output-stderr-env-canary'


def envelope(action=None, **overrides):
    value = {'type':'result','subtype':'success','is_error':False,
        'structured_output':{'action':action or {'kind':'inspect_frames','run_id':RUN,
            'base_version_id':BASE,'asset_id':ASSET,'version_id':BASE,'timestamps_sec':[1]}}}
    value.update(overrides)
    return value


def failure(raw):
    with pytest.raises(planner.PlannerError) as caught:
        planner._agent_reply(raw,AgentStepContext(context=context(),step_index=0))
    return caught.value


def wire(*records):
    return '\n'.join(json.dumps(record) for record in records).encode()


@pytest.mark.parametrize('raw,expected,cause',[
    (SECRET.encode(),'action_transport_invalid','value'),
    (wire(envelope(structured_output={'action':{'kind':SECRET}})),'action_schema_invalid','schema'),
    (wire(envelope(action={'kind':'inspect_frames','run_id':OTHER,'base_version_id':BASE,
        'asset_id':ASSET,'version_id':BASE,'timestamps_sec':[1]})),'action_scope_invalid','value'),
    (wire(envelope(action={'kind':'inspect_frames','run_id':RUN,'base_version_id':BASE,
        'asset_id':OTHER,'version_id':BASE,'timestamps_sec':[1]})),'action_domain_invalid','value'),
    (wire({'type':'system','tools':[SECRET]},envelope()),'action_runtime_event_invalid','value'),
    (wire(envelope(is_error=True,error=SECRET)),'action_terminal_invalid','value'),
    (wire(envelope(subtype='error_max_turns',errors=[SECRET])),'native_max_turns','value'),
    (wire(envelope(subtype='error_max_structured_output_retries',errors=[SECRET])),'native_structured_output_retries','value'),
    (wire(envelope(subtype='error_max_budget_usd',errors=[SECRET])),'native_budget_limit','value'),
    (wire(envelope(stop_reason='max_tokens',errors=[SECRET])),'native_token_limit','value'),
    (wire({'type':'rate_limit_event','rate_limit_info':{'status':'rejected','message':SECRET}},
        envelope(is_error=True,errors=[SECRET])),'native_usage_rejected','value'),
])
def test_invalid_actions_keep_rejection_and_report_only_fixed_code(raw,expected,cause,caplog):
    error = failure(raw)
    assert planner.planner_failure_details(error) == (expected,cause)
    with caplog.at_level(logging.WARNING):
        _log_planner_failure(TURN,error)
    assert f'code={expected} cause={cause}' in caplog.text
    assert SECRET not in caplog.text and 'Traceback' not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_unknown_native_strings_never_become_diagnostic_labels(caplog):
    error = failure(wire(envelope(subtype=SECRET,error=SECRET,errors=[SECRET])))
    assert error.failure_code == 'action_terminal_invalid'
    with caplog.at_level(logging.WARNING):
        _log_planner_failure(SECRET+'\n',error)
        _log_planner_failure(TURN.upper(),planner.PlannerError(SECRET,failure_code=SECRET,cause_category=SECRET))
        _log_planner_failure(TURN,RuntimeError(SECRET))
    assert SECRET not in caplog.text
    assert 'invocation=invalid_invocation' in caplog.text
    assert 'code=unexpected_exception cause=other' in caplog.text


def test_mutated_error_subclass_labels_are_sanitized(caplog):
    class InjectedError(planner.PlannerError):
        def __str__(self):
            raise AssertionError('diagnostics must never render exceptions')
    error = InjectedError(SECRET)
    error.failure_code = SECRET
    error.cause_category = SECRET
    with caplog.at_level(logging.WARNING):
        _log_planner_failure(TURN,error)
    assert 'code=planner_error cause=other' in caplog.text and SECRET not in caplog.text


def test_usage_warning_does_not_reject_valid_action():
    step = AgentStepContext(context=context(),step_index=0)
    warning = {'type':'rate_limit_event','rate_limit_info':{'status':'rejected','message':SECRET}}
    action = planner._agent_reply(wire(warning,envelope()),step)
    assert action.kind == 'inspect_frames'


@pytest.mark.parametrize('error,code,expected_state,expected_error',[
    (planner.PlannerError(SECRET,failure_code='action_schema_invalid',cause_category='schema'),'action_schema_invalid','failed','planner_failed'),
    (planner.PlannerTimeout(SECRET),'timeout','failed','timeout'),
    (planner.PlannerCancelled(SECRET),'cancelled','cancelled','cancelled'),
    (RuntimeError(SECRET),'unexpected_exception','failed','planner_failed'),
])
def test_broker_logs_fixed_categories_preserving_terminal_protocol(setup_broker,caplog,error,code,expected_state,expected_error):
    make,_ = setup_broker
    broker,injected = make()
    injected.error = error
    with caplog.at_level(logging.WARNING):
        broker.submit(TURN,context())
        result = terminal(broker)
    assert result.state == expected_state and result.error_code == expected_error
    assert f'invocation={TURN} code={code}' in caplog.text
    assert SECRET not in caplog.text and 'Traceback' not in caplog.text
    assert result.reply is None


def test_configuration_and_input_rejections_are_distinct(monkeypatch):
    def reject(_):
        raise planner.PlannerError(SECRET)
    monkeypatch.setattr(planner,'_configuration',reject)
    with pytest.raises(planner.PlannerError) as caught:
        planner.ClaudePlanner().plan_step(AgentStepContext(context=context(),step_index=0))
    assert caught.value.failure_code == 'configuration_failed'
    monkeypatch.setattr(planner,'_configuration',lambda _: (None,None,None))
    monkeypatch.setattr(planner,'_agent_input',lambda _: (_ for _ in ()).throw(ValueError(SECRET)))
    with pytest.raises(planner.PlannerError) as caught:
        planner.ClaudePlanner().plan_step(AgentStepContext(context=context(),step_index=0))
    assert caught.value.failure_code == 'input_invalid_or_large'


@pytest.mark.parametrize('body,expected',[
    ("sys.stdin.buffer.read();sys.stderr.write(%r);sys.exit(3)" % SECRET,'execute_nonzero'),
    ("sys.stdin.buffer.read();sys.stdout.write(%r*300000)" % SECRET,'execute_output_limit'),
])
def test_offline_process_failures_do_not_log_private_streams(fake,body,expected,caplog):
    with pytest.raises(planner.PlannerError) as caught:
        planner.ClaudePlanner(fake(body)).plan(context())
    assert caught.value.failure_code == expected
    with caplog.at_level(logging.WARNING):
        _log_planner_failure(TURN,caught.value)
    assert SECRET not in caplog.text and f'code={expected}' in caplog.text


def test_logging_failure_cannot_strand_broker_terminal_update(setup_broker,monkeypatch):
    logger = logging.getLogger('app.assistant_broker')
    def broken_handler(*args,**kwargs):
        raise OSError(SECRET)
    monkeypatch.setattr(logger,'warning',broken_handler)
    make,_ = setup_broker
    broker,injected = make()
    injected.error = planner.PlannerError(SECRET)
    broker.submit(TURN,context())
    result = terminal(broker)
    assert result.state == 'failed' and result.error_code == 'planner_failed'


def test_specific_nested_planner_failure_code_survives_valueerror_catch(monkeypatch):
    import app.assistant_agent_contracts as contracts
    expected = planner.PlannerError(SECRET,failure_code='execute_output_limit',cause_category='type')
    def reject(*args):
        raise expected
    monkeypatch.setattr(contracts,'validate_agent_action',reject)
    error = failure(wire(envelope()))
    assert error is expected
    assert planner.planner_failure_details(error) == ('execute_output_limit','type')
