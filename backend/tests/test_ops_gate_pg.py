"""AICP-350 deploy admission gate on REAL PostgreSQL row locks
(docs/assistant-drain-check-design.md §2-§3).

Opt in with RESKIN_TEST_PG_URL pointing at a THROWAWAY database (never
production), e.g.::

    docker run --rm -d --name reskin-opsgate-pgtest -e POSTGRES_PASSWORD=t \\
        -p 127.0.0.1:15456:5432 postgres:16-alpine
    RESKIN_TEST_PG_URL=postgresql://postgres:t@127.0.0.1:15456/postgres pytest tests/test_ops_gate_pg.py

Each test gets its own schema with the full metadata. The ``deploy/ops-gate``
SQL files are executed verbatim (psql ``:'var'`` placeholders become bound
parameters); those tests skip when the repo root is not mounted.

What is proven here:
* ``begin.sql`` WAITS for an admission holding the gate ``FOR SHARE`` and every
  admission after it is refused — no check-then-launch window.
* begin refuses while paused; ``end`` is a CAS on owner (+ generation);
  ``force-end`` is a CAS on generation.
* a duplicate ``Idempotency-Key`` waits for the first transaction: replay after
  its commit, fresh execution after its rollback.
* the lock order gate -> key -> spend day lock cannot deadlock with an ungated
  worker reserve or with a concurrent ``begin``.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

_OPT_IN = "RESKIN_TEST_PG_URL"
pytestmark = [
    pytest.mark.skipif(not os.environ.get(_OPT_IN),
                       reason=f"opt in with {_OPT_IN}; throwaway test DB only"),
    pytest.mark.spend_cap,
]

_GATE_SQL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "ops-gate")


@pytest.fixture
def pg():
    from app.db import Base
    import app.models  # noqa: F401 - registers every table (incl. ops_gate)
    from app.ops_gate import OpsAdmissionGate

    url = os.environ[_OPT_IN]
    assert "re-skin-db" not in url and "@db" not in url, "never the production database"
    schema = "og_" + uuid.uuid4().hex[:12]
    admin = create_engine(url, poolclass=NullPool)
    with admin.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, poolclass=NullPool, connect_args={
        "options": f"-c search_path={schema} -c lock_timeout=20000"})
    Base.metadata.create_all(engine)
    SF = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    with SF.begin() as db:
        db.add(OpsAdmissionGate(id=1, paused=False, generation=0))
    try:
        yield SF
    finally:
        engine.dispose()
        with admin.begin() as c:
            c.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _gate_sql(name: str, **params) -> str:
    """Split deploy/ops-gate/<name>.sql into statements + a binder that turns
    psql ``:'var'`` placeholders into bound parameters."""
    path = os.path.join(_GATE_SQL_DIR, f"{name}.sql")
    if not os.path.exists(path):
        pytest.skip("deploy/ops-gate not mounted (run with the repo root)")
    raw = open(path).read()
    body = "\n".join(l for l in raw.splitlines()
                     if not l.lstrip().startswith("--") and not l.lstrip().startswith("\\"))
    stmts = [s.strip() for s in body.split(";") if s.strip()]

    def bind(stmt):
        names = re.findall(r":'([a-z_]+)'", stmt)
        return re.sub(r":'([a-z_]+)'", lambda m: f":{m.group(1)}", stmt), {n: params[n] for n in names}

    return stmts, bind


def run_gate_sql(SF, name: str, **params) -> str:
    stmts, bind = _gate_sql(name, **params)
    out = None
    with SF() as s:
        for st in stmts:
            q, p = bind(st)
            res = s.execute(text(q), p)
            if res.returns_rows:
                row = res.first()  # psql -At -F' ' prints the columns space-joined
                out = None if row is None else " ".join(str(c) for c in row)
        s.commit()
    return out


def _in_thread(fn):
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        box["at"] = time.monotonic()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


# ---------------------------------------------------------------------------
# begin / end / force-end
# ---------------------------------------------------------------------------


def test_begin_waits_for_an_inflight_admission_then_fences_every_later_one(pg):
    from app import ops_gate

    a = pg()
    ops_gate.lock_admitting(a)  # an admission in flight holds the share lock
    t, box = _in_thread(lambda: run_gate_sql(pg, "begin", owner="deploy:t1", reason="test"))
    time.sleep(1.0)
    assert t.is_alive(), "begin must wait for the admission's share lock"
    committed_at = time.monotonic()
    a.commit()
    a.close()
    t.join(10)
    assert "error" not in box, box.get("error")
    assert box["value"] == "begun 1"
    assert box["at"] >= committed_at
    b = pg()
    with pytest.raises(ops_gate.AdmissionPaused):
        ops_gate.lock_admitting(b)
    b.close()


def test_begin_refuses_when_paused_and_end_is_a_cas(pg):
    assert run_gate_sql(pg, "begin", owner="deploy:A", reason="") == "begun 1"
    assert run_gate_sql(pg, "begin", owner="deploy:B", reason="x") == "refused 1 deploy:A"
    assert run_gate_sql(pg, "end", owner="deploy:B", gen="") == "not-owner"
    assert run_gate_sql(pg, "end", owner="deploy:A", gen="7") == "not-owner"
    assert run_gate_sql(pg, "status").startswith("paused")
    assert run_gate_sql(pg, "end", owner="deploy:A", gen="1") == "ended 1"
    assert run_gate_sql(pg, "end", owner="deploy:A", gen="1") == "not-owner"
    # A lost begin output: end by the owner token alone (empty gen).
    assert run_gate_sql(pg, "begin", owner="deploy:C", reason="") == "begun 2"
    assert run_gate_sql(pg, "end", owner="deploy:C", gen="") == "ended 2"
    assert run_gate_sql(pg, "status").startswith("admitting")


def test_force_end_is_a_cas_on_generation(pg):
    assert run_gate_sql(pg, "begin", owner="deploy:dead", reason="") == "begun 1"
    assert run_gate_sql(pg, "force-end", gen="2") == "no-match"
    assert run_gate_sql(pg, "force-end", gen="1") == "force-ended 1"
    # The dead deploy's trap firing later cannot reopen a NEWER pause.
    assert run_gate_sql(pg, "begin", owner="deploy:new", reason="") == "begun 2"
    assert run_gate_sql(pg, "end", owner="deploy:dead", gen="1") == "not-owner"
    assert run_gate_sql(pg, "status").startswith("paused 2")


# ---------------------------------------------------------------------------
# Idempotency-Key
# ---------------------------------------------------------------------------


def _idem(key="k1", sha="a" * 64):
    from app.ops_gate import IdemRequest

    return IdemRequest(key=key, route="POST /x", request_sha256=sha)


def test_duplicate_key_waits_then_replays_after_commit(pg):
    from app import ops_gate

    a = pg()
    ops_gate.lock_admitting(a)
    adm_a = ops_gate.claim_idempotency_key(a, _idem())
    assert adm_a.replay is None

    def second():
        b = pg()
        try:
            ops_gate.lock_admitting(b)
            adm = ops_gate.claim_idempotency_key(b, _idem())
            b.commit()
            return adm.replay
        finally:
            b.close()

    t, box = _in_thread(second)
    time.sleep(1.0)
    assert t.is_alive(), "a duplicate key must wait for the first transaction"
    adm_a.record("run_caption", ["c-1"])
    a.commit()
    a.close()
    t.join(10)
    assert "error" not in box, box.get("error")
    assert box["value"] == {"resource_kind": "run_caption", "resource_ids": ["c-1"]}


def test_duplicate_key_executes_afresh_after_rollback(pg):
    from app import ops_gate

    a = pg()
    ops_gate.lock_admitting(a)
    ops_gate.claim_idempotency_key(a, _idem("k2"))

    def second():
        b = pg()
        try:
            ops_gate.lock_admitting(b)
            adm = ops_gate.claim_idempotency_key(b, _idem("k2"))
            adm.record("run_caption", ["c-2"])
            b.commit()
            return adm.replay
        finally:
            b.close()

    t, box = _in_thread(second)
    time.sleep(1.0)
    assert t.is_alive()
    a.rollback()  # validation failed / killed: the key goes with everything else
    a.close()
    t.join(10)
    assert "error" not in box, box.get("error")
    assert box["value"] is None  # B owned the key and executed
    with pg() as s:
        row = s.get(ops_gate.ApiIdempotencyKey, "k2")
        assert row.operation["resource_ids"] == ["c-2"]


def test_duplicate_key_with_a_different_request_is_409_after_commit(pg):
    from fastapi import HTTPException

    from app import ops_gate

    with pg() as a:
        ops_gate.lock_admitting(a)
        ops_gate.claim_idempotency_key(a, _idem("k3")).record("x", ["1"])
        a.commit()
    with pg() as b:
        ops_gate.lock_admitting(b)
        with pytest.raises(HTTPException) as exc:
            ops_gate.claim_idempotency_key(b, _idem("k3", sha="b" * 64))
        assert exc.value.status_code == 409


# ---------------------------------------------------------------------------
# Lock order: gate -> key -> spend day lock (design §3)
# ---------------------------------------------------------------------------


def test_lock_order_gate_key_reserve_against_worker_reserve_and_begin(pg, monkeypatch):
    """A keyed admission (gate share -> key -> day lock) blocked behind an
    ungated worker reserve holding the day lock, with a deploy `begin` queued
    behind the admission: no deadlock; the worker commits, then the admission,
    then begin; the next admission is refused."""
    from app import ops_gate, spend
    from app.config import settings

    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 1000.0)
    now = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
    order: list[str] = []

    w = pg()  # worker: downstream reserve, never touches the gate
    spend.reserve(w, "run:worker", 10.0, kind="worker draw", now=now)

    def admission():
        s = pg()
        try:
            ops_gate.lock_admitting(s)
            adm = ops_gate.claim_idempotency_key(s, _idem("k-order"))
            adm.record("run_caption", ["c-o"])
            spend.reserve(s, "caption:c-o", 5.0, kind="captions", now=now)  # waits on day lock
            order.append(("admission-commit", time.monotonic()))
            s.commit()
        finally:
            s.close()

    ta, ba = _in_thread(admission)
    time.sleep(0.8)
    assert ta.is_alive(), "the admission waits on the worker's day lock"

    def begin():
        return run_gate_sql(pg, "begin", owner="deploy:order", reason="")

    tb, bb = _in_thread(begin)
    time.sleep(0.8)
    assert tb.is_alive(), "begin waits on the admission's share lock"
    worker_commit = time.monotonic()
    w.commit()
    w.close()
    ta.join(15)
    tb.join(15)
    assert "error" not in ba, ba.get("error")
    assert "error" not in bb, bb.get("error")
    assert bb["value"] == "begun 1"
    admission_commit = dict(order)["admission-commit"]
    assert admission_commit >= worker_commit   # the admission waited for the day lock
    assert bb["at"] >= admission_commit         # begin waited for the admission's commit
    with pg() as s:
        with pytest.raises(ops_gate.AdmissionPaused):
            ops_gate.lock_admitting(s)


def test_autopilot_gate_share_lock_is_fenced_by_begin(pg):
    """The autopilot's admission (core._gate) takes the global gate share lock
    first: begin waits for it, and the next one reads held."""
    from app.autopilot import core
    from app.models_autopilot import AutopilotGate

    with pg.begin() as s:
        s.add(AutopilotGate(id=1, held=False, generation=0))
    a = pg()
    assert core._gate(a, share=True).held is False
    t, box = _in_thread(lambda: run_gate_sql(pg, "begin", owner="deploy:ap", reason=""))
    time.sleep(1.0)
    assert t.is_alive()
    a.commit()
    a.close()
    t.join(10)
    assert box.get("value") == "begun 1"
    with pg() as s:
        assert core._gate(s, share=True).held is True
        s.rollback()


def test_running_uncounted_job_holds_the_gate_until_it_ends(pg):
    """Drive re-link / caption-placement scan jobs have no drain row: their
    job_barrier holds the gate share lock for the whole run, so begin waits
    for the job; a job starting after begin is told to defer."""
    from contextlib import contextmanager

    from app import ops_gate

    @contextmanager
    def gs():
        s = pg()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    started, release = threading.Event(), threading.Event()

    def job():
        with ops_gate.job_barrier(gs) as ok:
            assert ok
            started.set()
            release.wait(10)  # the job's real work
        return "done"

    tj, bj = _in_thread(job)
    assert started.wait(5)
    tb, bb = _in_thread(lambda: run_gate_sql(pg, "begin", owner="deploy:job", reason=""))
    time.sleep(1.0)
    assert tb.is_alive(), "begin must wait for the running job"
    done_at = time.monotonic()
    release.set()
    tj.join(10)
    tb.join(10)
    assert bj.get("value") == "done" and bb.get("value") == "begun 1"
    assert bb["at"] >= done_at
    with ops_gate.job_barrier(gs) as ok:
        assert ok is False
