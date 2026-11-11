"""AICP-351 on PostgreSQL: the concurrency the attempt protocol relies on.

Opt-in like tests/test_ops_gate_pg.py: set RESKIN_TEST_PG_URL to a DISPOSABLE
PostgreSQL (this creates and drops a database).

* two senders racing ``paid_attempts.begin`` for one owner: exactly one gets a
  ``started`` row (the partial unique index), the other sends nothing;
* two executors racing ``LlmLedger.claim`` for the same call number of one
  attempt: exactly one permit (one send per (attempt, call_no));
* a sender's receipt racing recovery's park on the same row: the row ends in
  one coherent state (parked with the late receipt, or accepted).
"""
from __future__ import annotations

import os
import sys
import threading
import uuid
from contextlib import contextmanager

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    import app.models  # noqa: F401
    from app.db import Base

    name = "psa_" + uuid.uuid4().hex[:12]
    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    url = make_url(_PG_URL).set(database=name).render_as_string(hide_password=False)
    eng = create_engine(url, pool_size=10, max_overflow=0)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture(scope="module")
def sessions(engine):
    from sqlalchemy.orm import sessionmaker

    SF = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def cm():
        db = SF()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    cm.SF = SF
    return cm


def _race(fn, n=2):
    barrier = threading.Barrier(n)
    out, errs = [], []

    def run():
        barrier.wait()
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=run) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    return out, errs


def test_one_started_attempt_per_owner(sessions):
    from app import paid_attempts
    from app.paid_attempts_models import OWNER_RUN_SEGMENT

    owner = str(uuid.uuid4())
    out, errs = _race(lambda: paid_attempts.begin(
        OWNER_RUN_SEGMENT, owner, kind="video", run_id=None, request={"o": owner},
        credits=10.0, scope_key="run:pg", draw_kind="video", sessions=sessions))
    assert len(out) == 1
    assert len(errs) == 1 and isinstance(errs[0], paid_attempts.AttemptBlocked)
    with sessions() as db:
        assert paid_attempts.attempt_count(db, OWNER_RUN_SEGMENT, owner) == 1


def test_one_permit_per_call_number(sessions):
    from app import paid_attempts
    from app.paid_attempts_models import OWNER_LLM_TRANSCRIBE

    aid = paid_attempts.begin_llm(OWNER_LLM_TRANSCRIBE, str(uuid.uuid4()), max_calls=3,
                                  lease_sec=60, fence_token="t", sessions=sessions)
    ledgers = [paid_attempts.LlmLedger(aid, scope_key="pg-llm", kind="transcribe",
                                       sessions=sessions) for _ in range(4)]
    it = iter(ledgers)
    lock = threading.Lock()

    def claim():
        with lock:
            led = next(it)
        return led.claim()

    out, errs = _race(claim, n=4)
    assert out == [1]
    assert len(errs) == 3 and all(isinstance(e, paid_attempts.LedgerFenced) for e in errs)
    from app.paid_attempts_models import PaidSubmitAttempt

    with sessions() as db:
        assert db.get(PaidSubmitAttempt, aid).calls_sent == 1


def test_receipt_racing_park_is_coherent(sessions):
    from app import paid_attempts
    from app.paid_attempts_models import OWNER_RUN_SEGMENT, PaidSubmitAttempt

    owner = str(uuid.uuid4())
    aid = paid_attempts.begin(OWNER_RUN_SEGMENT, owner, kind="video", run_id=None,
                              request={}, credits=10.0, scope_key="run:pg2", draw_kind="video",
                              sessions=sessions)

    def receipt():
        with sessions() as db:
            return ("receipt", paid_attempts.record_outcome(
                db, aid, outcome="accepted", task_id="T-race", credits=10.0,
                scope_key="run:pg2", draw_kind="video"))

    def park():
        with sessions() as db:
            row = db.get(PaidSubmitAttempt, aid, with_for_update=True, populate_existing=True)
            return ("park", paid_attempts.park_started(db, row, reason="recovered"))

    barrier = threading.Barrier(2)
    res = []

    def run(fn):
        barrier.wait()
        res.append(fn())

    ts = [threading.Thread(target=run, args=(f,)) for f in (receipt, park)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    with sessions() as db:
        row = db.get(PaidSubmitAttempt, aid)
        got = dict(res)
        if got["receipt"]:           # the receipt won: accepted, park was a no-op
            assert row.status == "accepted" and got["park"] is False
        else:                        # the park won: the late receipt is kept on it
            assert row.status == "parked" and got["park"] is True
        assert row.provider_task_id == "T-race"
