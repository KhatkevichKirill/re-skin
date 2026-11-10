"""Bounded render configuration and long-job lease/cancel checks; no inference."""
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.test_assistant_turns import setup,service,accept,plan,NOW
from tests.test_assistant_supervisor import Broker
from app import assistant_executor as executor,assistant_media as media
from app.assistant_jobs import AssistantJob,accept_job,claim_job,request_cancel
from app.assistant_turns import AssistantTurnEdit
from app.assistant_supervisor import AssistantSupervisor
from app.config import _assistant_edit_timeout_sec


@pytest.mark.parametrize("raw,expected",[(None,600),("30",30),("450",450),("600",600)])
def test_render_timeout_environment_is_bounded_and_defaults_to_600(monkeypatch,raw,expected):
    if raw is None:
        monkeypatch.delenv("ASSISTANT_EDIT_TIMEOUT_SEC",raising=False)
    else:
        monkeypatch.setenv("ASSISTANT_EDIT_TIMEOUT_SEC",raw)
    assert _assistant_edit_timeout_sec()==expected


@pytest.mark.parametrize("raw",["29","601","-1","0","nan","600.5",""])
def test_invalid_render_timeout_environment_fails_closed(monkeypatch,raw):
    monkeypatch.setenv("ASSISTANT_EDIT_TIMEOUT_SEC",raw)
    with pytest.raises(ValueError,match=r"\[30,600\]"):
        _assistant_edit_timeout_sec()


def test_supervisor_passes_custom_edit_budget_without_extending_lease(setup):
    with setup.factory.begin() as session:
        turn=accept(setup,session)
        job=accept_job(session,plan(setup),request_scope="fixture",idempotency_key="render-budget",actor="operator",now=NOW)
        session.add(AssistantTurnEdit(id=str(uuid4()),turn_id=turn.id,job_id=job.id,idempotency_key="accept",created_at=NOW))
    calls=[]
    supervisor=AssistantSupervisor(setup.factory,service=service(setup),broker=Broker(setup),owner="supervisor",
        execute=lambda *args,**kwargs:calls.append(kwargs),clock=lambda:NOW,sleep=lambda _:None,edit_timeout_sec=450)
    supervisor.run_once()
    assert len(calls)==1
    assert calls[0]["timeout_sec"]==450 and calls[0]["lease_sec"]==120


@pytest.mark.parametrize("value",[0,29,601,True,600.0,"600"])
def test_supervisor_rejects_unbounded_or_wrong_type_edit_timeout(setup,value):
    with pytest.raises(ValueError,match="edit timeout"):
        AssistantSupervisor(setup.factory,service=service(setup),broker=Broker(setup),owner="supervisor",edit_timeout_sec=value)


def test_longer_job_renews_120_second_lease_and_remains_cancellable(setup,monkeypatch):
    now=datetime.now(timezone.utc)
    with setup.factory.begin() as session:
        job=accept_job(session,plan(setup),request_scope="fixture",idempotency_key="long-media",actor="operator",now=now)
        job_id=job.id
        lease=claim_job(session,job_id,owner="worker",now=now,lease_sec=120)
    ticks=[0.0]
    monkeypatch.setattr(executor,"time",SimpleNamespace(monotonic=lambda:ticks[0]))
    monkeypatch.setattr(executor,"_now",lambda:now+timedelta(seconds=ticks[0]))
    renewed=[]
    original=executor.renew_lease
    def renew(*args,**kwargs):
        renewed.append(kwargs["lease_sec"])
        return original(*args,**kwargs)
    monkeypatch.setattr(executor,"renew_lease",renew)
    guard=executor._Guard(setup.factory,lease,120,600)
    for tick in (31,62,93,124,155):
        ticks[0]=tick
        assert guard() is False
    assert renewed==[120]*5
    with setup.factory.begin() as session:
        assert session.get(AssistantJob,job_id).claim_count==1
        request_cancel(session,job_id,now=now+timedelta(seconds=156))
    ticks[0]=156
    with pytest.raises(media.AssistantMediaCancelled):
        guard.check()
    assert renewed==[120]*5
