"""
Tests for the Fal provider layer (app/pipeline_provider.py, AICP-312 Stage 1 /
AICP-330). SQLite, mocked Fal transport (a fake FalClient), no Redis.

Each AICP-330 checklist item has its tests here, named ``test_b1_*`` ..
``test_m5_*``; the PostgreSQL proofs (lock order, live migration) are in
test_pipeline_provider_pg.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import importlib.util
import re

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.orm import sessionmaker

from app import pipeline_provider as pp
from app import spend, storage
from app.config import settings
from app.db import Base
from app.fal_client import ResultOutcome, StatusOutcome, SubmitOutcome, request_urls
from app.models import KieSpendDay, KieSpendDraw, KieSpendReservation, Run, VideoProject
from app.provider_models import ProviderTask, ProviderTaskAction, ProviderTaskActionImmutable
from app.state_machine import (
    PROVIDER_TASK_ACTIVE_WHERE,
    PROVIDER_TASK_DRAIN_STATUSES,
    PROVIDER_TASK_NON_TERMINAL_STATUSES,
    PROVIDER_TASK_TRANSITIONS,
    ProjectStatus,
    ProviderTaskStatus as S,
    RunStatus,
    can_transition,
)

pytestmark = [pytest.mark.spend_cap, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)
REPO = os.path.join(os.path.dirname(__file__), "..", "..")
ENDPOINT = "minimax/h3-max/reframe-video"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sqlite_engine():
    """SQLite with real SAVEPOINT semantics (SQLAlchemy's documented pysqlite
    recipe): without it the outermost RELEASE SAVEPOINT commits, which would
    make admission's savepoint durable before its reservation. Prod is
    PostgreSQL; the PG proofs are in test_pipeline_provider_pg.py."""
    path = tempfile.mkstemp(suffix=".db")[1]
    eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        dbapi_conn.isolation_level = None
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    @event.listens_for(eng, "begin")
    def _begin(conn):
        conn.exec_driver_sql("BEGIN")

    return eng


@pytest.fixture(scope="module")
def engine():
    eng = _sqlite_engine()
    Base.metadata.create_all(eng)
    yield eng


class FakeClient:
    """Scripted Fal transport; records every call."""

    def __init__(self):
        self.calls = []
        self.submit_outcome = SubmitOutcome("accepted", 200, "req-1", request_urls("minimax/h3-max", "req-1"))
        self.status_outcomes = []
        self.result_outcome = ResultOutcome("ok", 200, body={"video": {"url": "https://v3.fal.media/out.mp4"}})
        self.cancel_code = 202
        self.download_bytes = b"video-bytes"
        self.download_error = None
        self.upload_error = None

    def close(self):
        pass

    def upload(self, data, **kw):
        self.calls.append(("upload", len(data)))
        if self.upload_error:
            raise self.upload_error
        return f"https://v3.fal.media/in-{len(self.calls)}.mp4"

    def submit(self, endpoint_id, body, **kw):
        self.calls.append(("submit", endpoint_id, dict(body)))
        out = self.submit_outcome
        if out.kind == "accepted" and out.request_id == "req-1" and self.count("submit") > 1:
            rid = f"req-{self.count('submit')}"
            return SubmitOutcome("accepted", 200, rid, request_urls("minimax/h3-max", rid))
        return out

    def status(self, url):
        self.calls.append(("status", url))
        return self.status_outcomes.pop(0) if self.status_outcomes else StatusOutcome("ok", 200, "IN_PROGRESS")

    def result(self, url):
        self.calls.append(("result", url))
        return self.result_outcome

    def cancel(self, url):
        self.calls.append(("cancel", url))
        return self.cancel_code

    def download(self, url, dst, *, max_bytes):
        self.calls.append(("download", url))
        if self.download_error:
            raise self.download_error
        with open(dst, "xb") as fh:
            fh.write(self.download_bytes)
        return len(self.download_bytes)

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)


@pytest.fixture()
def env(engine, monkeypatch):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

    @contextmanager
    def sessions():
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    client = FakeClient()
    enqueued = []
    monkeypatch.setattr(pp, "_sessions", sessions)
    monkeypatch.setattr(pp, "_client", lambda: client)
    monkeypatch.setattr(pp, "_enqueue", lambda fn, tid, delay_sec=0: enqueued.append((fn, tid, delay_sec)))
    monkeypatch.setattr(pp, "_probe_duration", lambda path: 5.0)
    monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)
    monkeypatch.setattr(settings, "FAL_ENABLED", True)
    monkeypatch.setattr(settings, "FAL_API_KEY", "test-key")
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 250.0)

    class Env:
        pass

    e = Env()
    e.factory, e.sessions, e.client, e.enqueued = factory, sessions, client, enqueued
    with sessions() as s:
        p = VideoProject(source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready)
        s.add(p)
        s.flush()
        r = Run(project_id=p.id, prompt="x", model="seedance", resolution="720p", status=RunStatus.done)
        s.add(r)
        s.flush()
        e.pid, e.rid = p.id, r.id
    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", e.pid)
    e.proj_dir = os.path.join(storage.data_dir_root(), "projects", e.pid)
    os.makedirs(os.path.join(e.proj_dir, "reframe"), exist_ok=True)
    os.makedirs(os.path.join(e.proj_dir, "runs", e.rid, "results"), exist_ok=True)
    e.src = os.path.join(e.proj_dir, "source.mp4")
    with open(e.src, "wb") as fh:
        fh.write(b"source-bytes")
    yield e
    s = factory()
    s.execute(text("DROP TRIGGER IF EXISTS provider_task_actions_no_delete"))
    for model in (ProviderTaskAction, ProviderTask, KieSpendDraw, KieSpendReservation, KieSpendDay):
        s.query(model).delete()
    s.commit()
    s.close()


_counter = [0]


def _admit(e, *, op=None, dest=None, slot="main", owner_id=None, run=False, **kw):
    _counter[0] += 1
    n = _counter[0]
    dest = dest or (os.path.join(e.proj_dir, "runs", e.rid, "results", f"out{n}.mp4") if run
                    else os.path.join(e.proj_dir, "reframe", f"out{n}.mp4"))
    with e.sessions() as s:
        task, created = pp.admit(
            s, endpoint_id=ENDPOINT, params={"aspect_ratio": "9:16"},
            inputs=[pp.TaskInput("video_url", e.src, "video/mp4")], output_sec=5.0,
            resolution="768p", owner_kind="reframe_job", owner_id=owner_id or f"job{n}",
            owner_project_id=e.pid, owner_run_id=e.rid if run else None,
            destination_path=dest, operation_key=op or f"reframe_job:job{n}:{slot}:1",
            owner_slot=slot, **kw)
        return task.id, created


def _task(e, tid) -> ProviderTask:
    with e.sessions() as s:
        t = s.get(ProviderTask, tid)
        s.expunge(t)
        return t


def _set(e, tid, **values):
    with e.sessions() as s:
        s.execute(text("UPDATE provider_tasks SET " + ", ".join(f"{k}=:{k}" for k in values)
                       + " WHERE id=:id"), {**values, "id": tid})


def _charged(e) -> float:
    with e.sessions() as s:
        return spend.today_summary(s)["charged_usd"]


def _draws(e, tid) -> list:
    with e.sessions() as s:
        return s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key == spend.provider_task_call_key(tid))).scalars().all()


def _submitted(e, **kw) -> str:
    tid, _ = _admit(e, **kw)
    pp.fal_submit(tid)
    assert _task(e, tid).status == "submitted"
    return tid


# ---------------------------------------------------------------------------
# State machine, CHECK, index, drain list
# ---------------------------------------------------------------------------


def test_transition_map_is_exactly_the_design():
    expected = {
        ("queued", "abandoned"), ("queued", "submit_started"), ("queued", "failed"),
        ("submit_started", "submitted"), ("submit_started", "accepted_unrecorded"),
        ("submit_started", "refused"), ("submit_started", "submit_unknown"),
        ("submit_unknown", "submitted"), ("submit_unknown", "abandoned"),
        ("accepted_unrecorded", "submitted"),
        ("submitted", "fetching"), ("submitted", "failed"), ("submitted", "cancelling"),
        ("submitted", "needs_reconciliation"),
        ("fetching", "done"), ("fetching", "refused"), ("fetching", "failed"),
        ("cancelling", "failed"), ("cancelling", "fetching"), ("cancelling", "needs_reconciliation"),
        ("needs_reconciliation", "fetching"), ("needs_reconciliation", "failed"),
        ("needs_reconciliation", "cancelling"),
        ("closing", "failed"), ("refused", "refused"), ("refused", "abandoned"),
    } | {(s, "closing") for s in ("submit_started", "submit_unknown", "accepted_unrecorded",
                                  "submitted", "fetching", "cancelling", "needs_reconciliation")}
    actual = {(a.value, b.value) for a, bs in PROVIDER_TASK_TRANSITIONS.items() for b in bs}
    assert actual == expected
    for a in S:
        for b in S:
            assert can_transition(a, b) == ((a.value, b.value) in expected)
    assert {s.value for s in PROVIDER_TASK_NON_TERMINAL_STATUSES} == {
        "queued", "submit_started", "accepted_unrecorded", "submitted", "fetching",
        "cancelling", "needs_reconciliation", "closing", "submit_unknown"}
    assert "submit_unknown" not in {s.value for s in PROVIDER_TASK_DRAIN_STATUSES}
    assert len(PROVIDER_TASK_DRAIN_STATUSES) == 8


def _migration():
    path = os.path.join(os.path.dirname(__file__), "..", "alembic", "versions",
                        "m3c9e5a1b427_fal_provider_tasks.py")
    spec = importlib.util.spec_from_file_location("fal_migration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_texts_match_models_and_drain_check():
    mig = _migration()
    assert mig.ACTIVE_WHERE == PROVIDER_TASK_ACTIVE_WHERE
    idx = next(i for i in ProviderTask.__table__.indexes
               if i.name == "uq_provider_tasks_active_per_owner_slot")
    assert str(idx.dialect_options["postgresql"]["where"]) == PROVIDER_TASK_ACTIVE_WHERE
    assert str(idx.dialect_options["sqlite"]["where"]) == PROVIDER_TASK_ACTIVE_WHERE
    sql = open(os.path.join(REPO, "deploy", "drain-check.sql")).read()
    # AICP-350: the drain check splits provider tasks into an `active` row
    # (waited on) and a `parked` row (needs_reconciliation + submit_unknown,
    # listed only); together they are the drain statuses plus submit_unknown.
    rows = [l for l in sql.splitlines() if "provider_tasks" in l and "SELECT" in l]
    by_state = {("parked" if "'parked'" in l else "active"):
                set(re.findall(r"'([a-z_]+)'", l.split("IN", 1)[1])) for l in rows}
    assert by_state["parked"] == {"needs_reconciliation", "submit_unknown"}
    assert by_state["active"] | (by_state["parked"] - {"submit_unknown"}) == {
        s.value for s in PROVIDER_TASK_DRAIN_STATUSES}
    assert not by_state["active"] & by_state["parked"]
    # The migration's columns are the model's columns.
    model_cols = {c.name for c in ProviderTask.__table__.columns}
    src = open(mig.__file__).read()
    mig_cols = set(re.findall(r'sa\.Column\("([a-z_0-9]+)"', src.split('"provider_tasks",', 1)[1]
                              .split(ACTIONS_SPLIT, 1)[0]))
    assert mig_cols == model_cols


ACTIONS_SPLIT = "op.create_index(\"uq_provider_tasks_active_per_owner_slot\""


def test_actions_table_is_insert_only(env):
    tid, _ = _admit(env)
    with env.sessions() as s:
        task = s.get(ProviderTask, tid)
        pp.abandon(s, tid, actor="ops", note="nothing at Fal")
    with env.sessions() as s:
        a = s.execute(select(ProviderTaskAction)).scalars().one()
        a.note = "edited"
        with pytest.raises(ProviderTaskActionImmutable):
            s.flush()
        s.rollback()
    with env.sessions() as s:
        a = s.execute(select(ProviderTaskAction)).scalars().one()
        s.delete(a)
        with pytest.raises(ProviderTaskActionImmutable):
            s.flush()
        s.rollback()


def test_migration_triggers_reject_update_and_delete_on_sqlite():
    eng = _sqlite_engine()
    Base.metadata.create_all(eng)
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    mig = _migration()
    with eng.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mig.downgrade()
            mig.upgrade()
        conn.execute(text(
            "INSERT INTO provider_tasks (id, provider, endpoint, kind, owner_kind, owner_id, "
            "owner_project_id, destination_path, operation_key, status, reserved_mcredits, "
            "request_spec, input_refs, request_sha256, output_sec, resolution, estimated_usd, "
            "version_id) VALUES ('t1','fal','e','k','o','1','p','/d','op','failed',1,'{}','[]',"
            "'h',1,'768p',1,0)"))
        conn.execute(text("INSERT INTO provider_task_actions (id, task_id, action, actor, "
                          "from_status, to_status) VALUES ('a1','t1','close','x','failed','failed')"))
        from sqlalchemy.exc import DatabaseError
        with pytest.raises(DatabaseError), conn.begin_nested():
            conn.execute(text("UPDATE provider_task_actions SET note='x'"))
        with pytest.raises(DatabaseError), conn.begin_nested():
            conn.execute(text("DELETE FROM provider_task_actions"))
        with Operations.context(MigrationContext.configure(conn)):
            with pytest.raises(RuntimeError, match="refusing to downgrade"):
                mig.downgrade()


# ---------------------------------------------------------------------------
# B2 — the provider column must not break kie.ai accounting
# ---------------------------------------------------------------------------


def test_b2_kie_draw_lands_after_the_migration_on_sqlite(monkeypatch):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    eng = _sqlite_engine()
    Base.metadata.create_all(eng)
    mig = _migration()
    with eng.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mig.downgrade()  # the spend tables as they are at l2b8d4f0a316
        assert "provider" not in {c["name"] for c in inspect(conn).get_columns("kie_spend_draws")}
        conn.execute(text("INSERT INTO kie_spend_reservations (id, scope_key, day, kind, "
                          "reserved_mcredits, consumed_mcredits, charged_mcredits) VALUES "
                          "('r0','run:old','2026-10-27','run_create',1000,0,1000)"))
        with Operations.context(MigrationContext.configure(conn)):
            mig.upgrade()
        # an OLD writer (does not name the column) after the migration
        conn.execute(text("INSERT INTO kie_spend_draws (id, call_key, reservation_id, kind, "
                          "estimated_mcredits, mcredits) VALUES ('d0','old-task','r0','video',5,5)"))
        assert conn.execute(text("SELECT provider FROM kie_spend_reservations")).scalar() == "kie"
        assert conn.execute(text("SELECT provider FROM kie_spend_draws")).scalar() == "kie"
    factory = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)

    @contextmanager
    def sessions():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    with sessions() as s:
        spend.reserve(s, "run:new", 100, kind="run_create")
        s.commit()
    spend.record_draw("run:new", 40, kind="video", call_key="kie-task-1", sessions=sessions)
    spend.settle_draw("kie-task-1", 30, scope_key="run:new", kind="video", sessions=sessions)
    with sessions() as s:
        d = s.execute(select(KieSpendDraw).where(KieSpendDraw.call_key == "kie-task-1")).scalar_one()
        assert (d.provider, d.mcredits) == ("kie", 30000)
        r = s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == "run:new")).scalar_one()
        assert r.provider == "kie" and r.consumed_mcredits == 30000


def test_b2_every_spend_insert_names_the_provider(env):
    seen = []

    @event.listens_for(env.factory.kw["bind"], "before_cursor_execute")
    def _spy(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("INSERT INTO KIE_SPEND_DRAWS") or \
                statement.lstrip().upper().startswith("INSERT INTO KIE_SPEND_RESERVATIONS"):
            seen.append(statement)

    try:
        spend.record_draw("run:x", 5, kind="video", call_key="k-b2", sessions=env.sessions)
        with env.sessions() as s:
            spend.reserve(s, "run:y", 5, kind="run_create")
        _submitted(env)
    finally:
        event.remove(env.factory.kw["bind"], "before_cursor_execute", _spy)
    assert seen and all("provider" in st for st in seen)
    with env.sessions() as s:
        assert {d.provider for d in s.execute(select(KieSpendDraw)).scalars()} == {"kie", "fal"}


# ---------------------------------------------------------------------------
# Admission (rule 4.1), M5, units
# ---------------------------------------------------------------------------


def test_admission_reserves_worst_case_in_credits_units(env):
    before = _charged(env)
    tid, created = _admit(env)
    t = _task(env, tid)
    assert created and t.status == "queued" and t.reservation_held
    # 5 s x 0.08 USD x overlap 2.0 = 0.80 USD, reserved as credits
    assert t.estimated_usd == pytest.approx(0.80)
    assert t.reserved_mcredits == round(0.80 / settings.KIE_USD_PER_CREDIT * 1000)
    assert _charged(env) - before == pytest.approx(0.80, abs=0.01)


def test_units_one_usd_raises_the_day_by_one_usd_mixed_with_kie(env, monkeypatch):
    spend.record_draw("run:k", 100, kind="video", call_key="kie-mixed", sessions=env.sessions)
    before = _charged(env)
    monkeypatch.setattr(pp.fal_models.FalEndpoint, "worst_case_usd", lambda self, r, s: 1.00)
    tid = _submitted(env)
    assert _charged(env) - before == pytest.approx(1.00, abs=0.01)
    assert len(_draws(env, tid)) == 1


def test_same_operation_key_returns_the_winner_and_reserves_nothing(env):
    tid, created = _admit(env, op="reframe_job:j:main:1", owner_id="j")
    before = _charged(env)
    tid2, created2 = _admit(env, op="reframe_job:j:main:1", owner_id="j")
    assert (tid2, created2) == (tid, False)
    assert _charged(env) == before


def test_active_slot_conflict_is_409_with_nothing_reserved(env):
    _admit(env, owner_id="same", op="reframe_job:same:main:1")
    before = _charged(env)
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env, owner_id="same", op="reframe_job:same:main:2")
    assert exc.value.status_code == 409 and _charged(env) == before


def test_m5_another_providers_task_in_the_slot_is_409_not_a_winner(env):
    tid, _ = _admit(env, owner_id="x5", op="reframe_job:x5:main:1")
    _set(env, tid, provider="byteplus")  # same operation_key string, other provider
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env, owner_id="x5", op="reframe_job:x5:main:1")
    assert exc.value.status_code == 409


def test_admission_refusals(env, monkeypatch):
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env, dest="/etc/passwd-out.mp4")
    assert exc.value.status_code == 422
    other = os.path.join(storage.data_dir_root(), "projects", "other-project", "x.mp4")
    with pytest.raises(pp.AdmissionRefused):
        _admit(env, dest=other)  # cross-project destination
    with pytest.raises(pp.AdmissionRefused):
        _admit(env, dest=os.path.join(env.proj_dir, "runs", env.rid, "results", "x.mp4"))  # project-scoped under runs/
    os.makedirs(os.path.join(env.proj_dir, "runs", env.rid, "clips"), exist_ok=True)
    with pytest.raises(pp.AdmissionRefused):
        _admit(env, run=True, dest=os.path.join(env.proj_dir, "runs", env.rid, "clips", "x.mp4"))
    monkeypatch.setattr(settings, "FAL_ENABLED", False)
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env)
    assert exc.value.status_code == 403


def test_admission_project_gate_allowlist_or_reframe(env, monkeypatch):
    """FAL_PROJECT_IDS admits any project; outside it only a reframe project."""
    from app.models import VideoProject

    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", "")
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env)  # the fixture project is face_swap
    assert exc.value.status_code == 403
    with env.sessions() as s:
        assert s.execute(select(ProviderTask)).first() is None
        s.get(VideoProject, env.pid).project_type = "reframe"
    _tid, created = _admit(env)
    assert created
    monkeypatch.setattr(settings, "FAL_ENABLED", False)  # still the master switch
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env)
    assert exc.value.status_code == 403
    assert pp.project_allowed("p1", "reframe") and pp.project_allowed("p1", None) is False
    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", "p1, p2")
    assert pp.project_allowed("p2", "face_swap")


def test_admission_refuses_untariffed_resolution_and_big_inputs(env, monkeypatch):
    with env.sessions() as s, pytest.raises(pp.AdmissionRefused, match="no tariff"):
        pp.admit(s, endpoint_id=ENDPOINT, params={}, inputs=[pp.TaskInput("video_url", env.src, "video/mp4")],
                 output_sec=5, resolution="1080p", owner_kind="k", owner_id="1",
                 owner_project_id=env.pid, destination_path=os.path.join(env.proj_dir, "reframe", "r1080.mp4"),
                 operation_key="k:1:main:1")
    monkeypatch.setattr(settings, "FAL_MAX_INPUT_BYTES", 4)
    with pytest.raises(pp.AdmissionRefused, match="FAL_MAX_INPUT_BYTES"):
        _admit(env)


def test_budget_exceeded_leaves_no_row(env, monkeypatch):
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.10)
    with pytest.raises(spend.BudgetExceeded):
        _admit(env, owner_id="budget")
    with env.sessions() as s:
        assert s.execute(select(ProviderTask).where(ProviderTask.owner_id == "budget")).first() is None


def test_admission_after_owner_deleted_is_refused(env):
    with env.sessions() as s:
        s.execute(text("DELETE FROM runs WHERE id=:r"), {"r": env.rid})
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env, run=True)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Submit (rules 4.3, 4.6, 4.7) — B3
# ---------------------------------------------------------------------------


def test_submit_accepted_draws_once_and_polls(env):
    tid = _submitted(env)
    t = _task(env, tid)
    assert t.request_id == "req-1" and not t.reservation_held and t.claim_owner is None
    assert t.status_url.endswith("/minimax/h3-max/requests/req-1/status")
    assert len(_draws(env, tid)) == 1 and _draws(env, tid)[0].provider == "fal"
    assert ("fal_poll", tid, settings.FAL_POLL_INTERVAL_SEC) in env.enqueued
    assert env.client.count("submit") == 1
    sent = next(c for c in env.client.calls if c[0] == "submit")[2]
    assert sent["video_url"].startswith("https://") and sent["resolution"] == "768p"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 429, 418])
def test_b3_every_submit_4xx_keeps_the_reservation(env, code):
    tid, _ = _admit(env)
    before = _charged(env)
    env.client.submit_outcome = SubmitOutcome("refused", code, error="nope")
    pp.fal_submit(tid)
    t = _task(env, tid)
    assert t.status == "refused" and t.reservation_held and t.error_code == f"submit_http_{code}"
    assert _charged(env) == before and _draws(env, tid) == []
    # still ACTIVE: holds the slot
    with env.sessions() as s:
        assert pp.active_task_for_project(s, env.pid) is not None


def test_b3_abandon_releases_and_close_converts(env):
    tid_a, _ = _admit(env)
    tid_c, _ = _admit(env)
    env.client.submit_outcome = SubmitOutcome("refused", 400, error="bad")
    pp.fal_submit(tid_a)
    pp.fal_submit(tid_c)
    before = _charged(env)
    with env.sessions() as s:
        pp.abandon(s, tid_a, actor="ops", note="dashboard shows nothing")
    assert _charged(env) == pytest.approx(before - 0.80, abs=0.01)
    with env.sessions() as s:
        pp.close(s, tid_c, actor="ops", note="unsure")
    t = _task(env, tid_c)
    assert t.status == "refused" and not t.reservation_held and len(_draws(env, tid_c)) == 1
    assert _charged(env) == pytest.approx(before - 0.80, abs=0.01)  # close never adds or releases
    with env.sessions() as s:
        assert [a.action for a in s.execute(select(ProviderTaskAction)).scalars()] == ["abandon", "close"]


@pytest.mark.parametrize("outcome", [
    SubmitOutcome("unknown", None, error="no response (ReadTimeout)"),
    SubmitOutcome("unknown", 502, error="bad gateway"),
    SubmitOutcome("unknown", 200, error="2xx without request_id"),
])
def test_submit_unknown_parks_with_reservation(env, outcome):
    tid, _ = _admit(env)
    env.client.submit_outcome = outcome
    pp.fal_submit(tid)
    t = _task(env, tid)
    assert t.status == "submit_unknown" and t.reservation_held and _draws(env, tid) == []


def test_local_failures_before_the_marker_release_and_never_post(env):
    tid, _ = _admit(env)
    with open(env.src, "ab") as fh:
        fh.write(b"changed")
    before = _charged(env)
    pp.fal_submit(tid)
    t = _task(env, tid)
    assert (t.status, t.error_code, t.reservation_held) == ("failed", "input_changed", False)
    assert env.client.count("submit") == 0 and _charged(env) == pytest.approx(before - 0.80, abs=0.01)


def test_upload_failure_is_local(env):
    from app.fal_client import FalUploadError

    tid, _ = _admit(env)
    env.client.upload_error = FalUploadError("cdn down")
    pp.fal_submit(tid)
    assert _task(env, tid).error_code == "upload_failed" and env.client.count("submit") == 0


def test_lost_lease_before_marker_sends_nothing(env, monkeypatch):
    tid, _ = _admit(env)
    real_upload = env.client.upload

    def steal(data, **kw):
        _set(env, tid, claim_owner="other-worker")  # another worker took an expired lease
        return real_upload(data, **kw)

    monkeypatch.setattr(env.client, "upload", steal)
    pp.fal_submit(tid)
    assert _task(env, tid).status == "queued" and env.client.count("submit") == 0


def test_two_workers_racing_the_claim_post_once(env):
    tid, _ = _admit(env)
    pp.fal_submit(tid)
    pp.fal_submit(tid)
    assert env.client.count("submit") == 1


def test_abandon_queued_with_live_claim_is_refused(env):
    tid, _ = _admit(env)
    _set(env, tid, claim_owner="w", claim_expires_at=_utcnow() + timedelta(minutes=5))
    with env.sessions() as s, pytest.raises(pp.ActionRefused):
        pp.abandon(s, tid, actor="ops", note="x")
    _set(env, tid, claim_owner=None, claim_expires_at=None)
    with env.sessions() as s:
        pp.abandon(s, tid, actor="ops", note="never sent")
    assert _task(env, tid).status == "abandoned"
    pp.fal_submit(tid)
    assert env.client.count("submit") == 0
    with env.sessions() as s, pytest.raises(pp.ActionRefused):
        pp.close(s, tid, actor="ops", note="x")


def test_acceptance_with_failing_draw_becomes_accepted_unrecorded_then_recovers(env, monkeypatch):
    tid, _ = _admit(env)
    real = spend.draw_in_session
    monkeypatch.setattr(spend, "draw_in_session", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
    pp.fal_submit(tid)
    t = _task(env, tid)
    assert t.status == "accepted_unrecorded" and t.request_id == "req-1" and t.reservation_held
    monkeypatch.setattr(spend, "draw_in_session", real)
    monkeypatch.setattr("app.recovery._live_run_ids", lambda conn, queue_names=None: set())
    pp.reconcile_provider_tasks(None)
    t = _task(env, tid)
    assert t.status == "submitted" and not t.reservation_held and len(_draws(env, tid)) == 1
    assert env.client.count("submit") == 1


def test_stale_marker_becomes_submit_unknown_never_reposted(env, monkeypatch):
    tid, _ = _admit(env)
    _set(env, tid, status="submit_started", claim_owner="dead",
         claim_expires_at=_utcnow() - timedelta(seconds=1))
    monkeypatch.setattr("app.recovery._live_run_ids", lambda conn, queue_names=None: set())
    pp.reconcile_provider_tasks(None)
    assert _task(env, tid).status == "submit_unknown"
    assert not any(fn == "fal_submit" for fn, _, _ in env.enqueued)


# ---------------------------------------------------------------------------
# Poll, cancel (rule 4.9) — M3
# ---------------------------------------------------------------------------


def test_poll_completed_moves_to_fetching_and_enqueues_fetch(env):
    tid = _submitted(env)
    env.client.status_outcomes = [StatusOutcome("ok", 200, "COMPLETED")]
    pp.fal_poll(tid)
    t = _task(env, tid)
    assert t.status == "fetching" and t.fetch_deadline_at is None and t.fetch_attempts == 0
    assert ("fal_fetch", tid, 0) in env.enqueued


def test_poll_auth_hold_and_404(env):
    tid = _submitted(env)
    env.client.status_outcomes = [StatusOutcome("auth", 401), StatusOutcome("ok", 200, "IN_QUEUE"),
                                  StatusOutcome("gone", 404)]
    pp.fal_poll(tid)
    assert (_task(env, tid).status, _task(env, tid).error_code) == ("submitted", "auth_rejected")
    pp.fal_poll(tid)
    assert _task(env, tid).error_code is None
    pp.fal_poll(tid)
    assert _task(env, tid).status == "needs_reconciliation"
    assert env.client.count("cancel") == 0


def test_poll_deadline_cancels_and_unconfirmed_cancel_reconciles(env):
    tid = _submitted(env)
    _set(env, tid, accepted_at=_utcnow() - timedelta(seconds=settings.FAL_POLL_DEADLINE_SEC + 1))
    pp.fal_poll(tid)
    assert _task(env, tid).status == "cancelling"
    assert ("fal_cancel", tid, 0) in env.enqueued
    _set(env, tid, cancel_requested_at=_utcnow() - timedelta(seconds=settings.FAL_CANCEL_CONFIRM_SEC + 1))
    pp.fal_poll(tid)
    assert _task(env, tid).status == "needs_reconciliation"
    assert len(_draws(env, tid)) == 1


@pytest.mark.parametrize("code,delivered", [(202, True), (400, True), (401, False), (429, False),
                                            (503, False), (None, False)])
def test_m3_cancel_delivery_is_durable(env, code, delivered):
    tid = _submitted(env)
    _set(env, tid, status="cancelling", cancel_requested_at=_utcnow())
    env.client.cancel_code = code
    pp.fal_cancel(tid)
    t = _task(env, tid)
    assert (t.cancel_delivered_at is not None) == delivered and t.claim_owner is None


def test_m3_recovery_reenqueues_a_lost_cancel_even_with_a_live_poll(env, monkeypatch):
    tid = _submitted(env)
    _set(env, tid, status="cancelling", cancel_requested_at=_utcnow())
    monkeypatch.setattr("app.recovery._live_run_ids", lambda conn, queue_names=None: {tid})
    env.enqueued.clear()
    pp.reconcile_provider_tasks(None)
    assert ("fal_cancel", tid, 0) in env.enqueued
    assert not any(fn == "fal_poll" for fn, _, _ in env.enqueued)  # the poll is live
    env.client.cancel_code = 202
    pp.fal_cancel(tid)
    env.enqueued.clear()
    pp.reconcile_provider_tasks(None)
    assert not any(fn == "fal_cancel" for fn, _, _ in env.enqueued)


def test_m3_cancel_does_nothing_once_the_row_moved_on(env):
    tid = _submitted(env)
    _set(env, tid, status="fetching")
    pp.fal_cancel(tid)
    assert env.client.count("cancel") == 0


# ---------------------------------------------------------------------------
# Fetch (rule 4.10) — B1
# ---------------------------------------------------------------------------


def _fetching(env):
    tid = _submitted(env)
    env.client.status_outcomes = [StatusOutcome("ok", 200, "COMPLETED")]
    pp.fal_poll(tid)
    return tid


def test_b1_a_fetch_that_starts_three_hours_late_still_downloads(env, monkeypatch):
    tid = _fetching(env)
    later = _utcnow() + timedelta(hours=3)
    monkeypatch.setattr(pp, "_now", lambda: later)
    pp.fal_fetch(tid)
    t = _task(env, tid)
    assert t.status == "done" and os.path.exists(t.destination_path)
    assert t.fetch_attempts == 1 and t.fetch_deadline_at == later + timedelta(seconds=settings.FAL_FETCH_WINDOW_SEC)


def test_b1_fal_jobs_use_their_own_queue_and_the_run_workers_do_not_serve_it(monkeypatch):
    from app import tasks

    calls = []

    class Q:
        def __init__(self, name):
            self.name = name

        def enqueue(self, target, *args, **kw):
            calls.append((self.name, target, args, kw))
            return type("J", (), {"id": "j"})()

        def enqueue_in(self, delay, target, *args, **kw):
            return self.enqueue(target, *args, **kw)

    monkeypatch.setattr(tasks, "_get_queue", lambda name=tasks.QUEUE_PROCESS: Q(name))
    for fn in tasks.FAL_JOBS:
        tasks.enqueue_fal_job(fn, "task-1", delay_sec=5 if fn == "fal_poll" else 0)
    assert {c[0] for c in calls} == {"fal"} and all(c[2] == ("task-1",) for c in calls)
    assert all(c[3]["job_timeout"] == (300 if c[1].endswith("fal_poll") else settings.FAL_JOB_TIMEOUT)
               for c in calls)
    src = open(os.path.join(REPO, "worker", "worker.py")).read()
    assert "QUEUE_FAL" not in src and '"fal"' not in src
    assert "QUEUE_FAL" in open(os.path.join(REPO, "worker", "fal_worker.py")).read()
    compose = open(os.path.join(REPO, "docker-compose.yml")).read()
    block = compose.split("  fal-worker:", 1)[1].split("\n  nginx:", 1)[0]
    assert 'profiles: ["fal"]' in block


def test_b1_live_run_ids_scans_the_fal_queue_when_asked(monkeypatch):
    from app import recovery

    seen = []

    class FakeQ:
        def __init__(self, name, connection=None):
            seen.append(name)
            self.job_ids = []

    class FakeReg(FakeQ):
        def get_job_ids(self):
            return []

        def cleanup(self):
            pass

    monkeypatch.setattr(recovery, "Queue", FakeQ)
    monkeypatch.setattr(recovery, "ScheduledJobRegistry", FakeReg)
    monkeypatch.setattr(recovery, "StartedJobRegistry", FakeReg)
    monkeypatch.setattr(recovery, "_live_worker_names", lambda conn: set())
    recovery._live_run_ids(None, queue_names=["fal"])
    assert set(seen) == {"fal"}
    seen.clear()
    recovery._live_run_ids(None)
    assert "fal" not in seen and "process" in seen


def test_b1_attempts_count_at_claim_so_crashed_attempts_exhaust(env):
    tid = _fetching(env)
    # five attempts that each died after taking the claim (counted at claim)
    _set(env, tid, fetch_attempts=settings.FAL_FETCH_MAX_ATTEMPTS,
         fetch_deadline_at=_utcnow() + timedelta(hours=1))
    pp.fal_fetch(tid)
    t = _task(env, tid)
    assert (t.status, t.error_code) == ("failed", "result_gone")
    assert len(_draws(env, tid)) == 1 and env.client.count("result") == 0


def test_b1_deadline_from_first_attempt_ends_result_gone(env, monkeypatch):
    tid = _fetching(env)
    env.client.result_outcome = ResultOutcome("retry", 503, error="busy")
    pp.fal_fetch(tid)
    t = _task(env, tid)
    assert t.status == "fetching" and t.fetch_attempts == 1 and t.claim_owner is None
    assert any(fn == "fal_fetch" and d == 60 for fn, _, d in env.enqueued)
    later = t.fetch_deadline_at + timedelta(seconds=1)
    monkeypatch.setattr(pp, "_now", lambda: later)
    pp.fal_fetch(tid)
    assert _task(env, tid).error_code == "result_gone"


def test_b1_installed_result_wins_over_exhaustion(env, monkeypatch):
    tid = _fetching(env)
    dest = _task(env, tid).destination_path
    with open(dest, "wb") as fh:  # a crash between os.replace and the done commit
        fh.write(b"video-bytes")
    _set(env, tid, fetch_attempts=settings.FAL_FETCH_MAX_ATTEMPTS)
    env.client.result_outcome = ResultOutcome("ok", 200, body={
        "video": {"url": "https://v3.fal.media/o.mp4"}, "placement": {"x": 1}})
    pp.fal_fetch(tid)
    t = _task(env, tid)
    assert t.status == "done" and env.client.count("download") == 0
    assert t.result_meta == {"placement": {"x": 1}}


def test_fetch_422_is_refused_and_the_draw_stays(env):
    tid = _fetching(env)
    env.client.result_outcome = ResultOutcome("refused", 422, category="moderation")
    pp.fal_fetch(tid)
    t = _task(env, tid)
    assert (t.status, t.error_code, t.reservation_held) == ("refused", "moderation", False)
    assert len(_draws(env, tid)) == 1


def test_fetch_rejects_wrong_duration_and_owner_gone(env, monkeypatch):
    tid = _fetching(env)
    monkeypatch.setattr(pp, "_probe_duration", lambda p: 2.0)
    pp.fal_fetch(tid)
    t = _task(env, tid)
    assert t.status == "fetching" and "duration" in t.error_message
    assert not [f for f in os.listdir(os.path.dirname(t.destination_path)) if f.endswith(".part")]
    monkeypatch.setattr(pp, "_probe_duration", lambda p: 5.0)
    tid2 = _fetching(env)
    os.rename(os.path.dirname(_task(env, tid2).destination_path),
              os.path.dirname(_task(env, tid2).destination_path) + "-gone")
    try:
        monkeypatch.setattr(pp, "_probe_duration", lambda p: 5.0)
        pp.fal_fetch(tid2)
        assert _task(env, tid2).error_code == "owner_gone" and len(_draws(env, tid2)) == 1
    finally:
        os.rename(os.path.dirname(_task(env, tid2).destination_path) + "-gone",
                  os.path.dirname(_task(env, tid2).destination_path))


# ---------------------------------------------------------------------------
# Close, backfill (M1, M4), adopt, resume
# ---------------------------------------------------------------------------


def test_close_of_submitted_cancels_and_fails_with_one_draw(env):
    tid = _submitted(env)
    with env.sessions() as s:
        pp.close(s, tid, actor="ops", note="giving up")
    assert _task(env, tid).status == "closing" and ("fal_close", tid, 0) in env.enqueued
    pp.fal_close(tid)
    t = _task(env, tid)
    assert (t.status, t.error_code) == ("failed", "operator_closed")
    assert env.client.count("cancel") == 1 and len(_draws(env, tid)) == 1


def test_m4_close_with_failing_draw_sets_pending_and_backfill_draws_once(env, monkeypatch):
    tid, _ = _admit(env)
    env.client.submit_outcome = SubmitOutcome("unknown", None, error="timeout")
    pp.fal_submit(tid)
    before = _charged(env)
    with env.sessions() as s:
        pp.close(s, tid, actor="ops", note="cannot tell")
    real = spend.draw_in_session
    monkeypatch.setattr(spend, "draw_in_session", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
    pp.fal_close(tid)
    t = _task(env, tid)
    assert (t.status, t.draw_pending, t.reservation_held) == ("failed", True, True)
    assert env.client.count("cancel") == 0  # no request id: nothing to cancel
    assert _charged(env) == before
    monkeypatch.setattr(spend, "draw_in_session", real)
    monkeypatch.setattr("app.recovery._live_run_ids", lambda conn, queue_names=None: set())
    pp.reconcile_provider_tasks(None)
    pp.reconcile_provider_tasks(None)
    t = _task(env, tid)
    assert (t.draw_pending, t.reservation_held) == (False, False) and len(_draws(env, tid)) == 1
    assert _charged(env) == before


def test_m1_backfill_never_charges_a_released_row(env, monkeypatch):
    tid, _ = _admit(env)
    env.client.submit_outcome = SubmitOutcome("refused", 422, error="bad")
    pp.fal_submit(tid)
    with env.sessions() as s:
        pp.abandon(s, tid, actor="ops", note="nothing exists")
    _set(env, tid, request_id="req-abandoned")  # even with a request id on the row
    monkeypatch.setattr("app.recovery._live_run_ids", lambda conn, queue_names=None: set())
    before = _charged(env)
    pp.reconcile_provider_tasks(None)
    assert _draws(env, tid) == [] and _charged(env) == before


def test_close_is_rejected_under_a_live_claim(env):
    tid = _fetching(env)
    _set(env, tid, claim_owner="fetcher", claim_expires_at=_utcnow() + timedelta(minutes=5))
    with env.sessions() as s, pytest.raises(pp.ActionRefused):
        pp.close(s, tid, actor="ops", note="x")


def test_adopt_verifies_records_and_draws(env):
    tid, _ = _admit(env)
    env.client.submit_outcome = SubmitOutcome("unknown", 502, error="gw")
    pp.fal_submit(tid)
    env.client.status_outcomes = [StatusOutcome("gone", 404)]
    with env.sessions() as s, pytest.raises(pp.ActionRefused):
        pp.adopt(s, tid, "req-zzz", actor="ops", note="seen on dashboard", client=env.client)
    assert _task(env, tid).status == "submit_unknown"
    env.client.status_outcomes = [StatusOutcome("ok", 200, "IN_PROGRESS")]
    with env.sessions() as s:
        pp.adopt(s, tid, "req-zzz", actor="ops", note="seen on dashboard", client=env.client)
    t = _task(env, tid)
    assert t.status == "submitted" and t.request_id == "req-zzz" and len(_draws(env, tid)) == 1
    assert t.status_url.endswith("/minimax/h3-max/requests/req-zzz/status")
    with env.sessions() as s:
        actions = s.execute(select(ProviderTaskAction).where(ProviderTaskAction.task_id == tid)
                            .order_by(ProviderTaskAction.at)).scalars().all()
        assert [(a.from_status, a.to_status) for a in actions] == [
            ("submit_unknown", "submit_unknown"), ("submit_unknown", "submitted")]


def test_second_adopt_of_the_same_id_is_409(env):
    a = _submitted(env)
    b, _ = _admit(env)
    env.client.submit_outcome = SubmitOutcome("unknown", 502)
    pp.fal_submit(b)
    with env.sessions() as s, pytest.raises(pp.ActionRefused) as exc:
        pp.adopt(s, b, "req-1", actor="ops", note="x", client=env.client)
    assert exc.value.status_code == 409


def test_resume_sends_a_fresh_cancel(env):
    tid = _submitted(env)
    _set(env, tid, status="needs_reconciliation")
    with env.sessions() as s:
        pp.resume(s, tid, actor="ops", note="retry cancel")
    t = _task(env, tid)
    assert t.status == "cancelling" and t.cancel_delivered_at is None
    assert ("fal_cancel", tid, 0) in env.enqueued


def test_recovery_never_posts(env, monkeypatch):
    monkeypatch.setattr("app.recovery._live_run_ids", lambda conn, queue_names=None: set())
    q, _ = _admit(env)
    _set(env, q, updated_at=_utcnow() - timedelta(seconds=settings.FAL_QUEUED_GRACE_SEC + 5))
    pp.reconcile_provider_tasks(None)
    assert ("fal_submit", q, 0) in env.enqueued and env.client.count("submit") == 0


# ---------------------------------------------------------------------------
# M2 — the delete paths
# ---------------------------------------------------------------------------


def test_m2_delete_project_and_run_409_on_an_active_task(env):
    from app import api_v2

    tid, _ = _admit(env, run=True)
    with env.sessions() as s, pytest.raises(HTTPException) as exc:
        api_v2.delete_run(env.rid, s)
    assert exc.value.status_code == 409
    with env.sessions() as s, pytest.raises(HTTPException) as exc:
        api_v2.delete_project(env.pid, s)
    assert exc.value.status_code == 409
    # a destination under the run's dir counts even when owner_run_id is unset
    _set(env, tid, owner_run_id=None)
    with env.sessions() as s, pytest.raises(HTTPException):
        api_v2.delete_run(env.rid, s)
    with env.sessions() as s:
        pp.abandon(s, tid, actor="ops", note="never sent")
    with env.sessions() as s:
        assert pp.active_task_for_run(s, env.rid, env.pid) is None


def test_m2_owner_id_naming_the_run_counts(env):
    tid, _ = _admit(env, owner_id="x-run")
    _set(env, tid, owner_id=env.rid)
    with env.sessions() as s:
        assert pp.active_task_for_run(s, env.rid, env.pid).id == tid


# ---------------------------------------------------------------------------
# Settings validation and the supervisor wiring
# ---------------------------------------------------------------------------


def test_default_settings_are_valid_and_bounds_are_checked(monkeypatch):
    assert pp.validate_settings() == []
    monkeypatch.setattr(settings, "FAL_CLAIM_LEASE_SEC", settings.FAL_JOB_TIMEOUT)
    monkeypatch.setattr(settings, "FAL_OUTPUT_LIFECYCLE_SEC", 3600)
    monkeypatch.setattr(settings, "FAL_INPUT_LIFECYCLE_SEC", 3600)
    assert len(pp.validate_settings()) == 3


def test_fal_ticker_runs_recovery_under_a_lease(monkeypatch):
    sys.path.insert(0, REPO)
    from worker import fal_ticker

    calls = []
    monkeypatch.setattr(pp, "reconcile_provider_tasks", lambda conn: calls.append("r") or {})
    monkeypatch.setattr(pp, "watchdog", lambda conn: calls.append("w") or [])

    class R:
        def __init__(self):
            self.kv = {}

        def set(self, k, v, nx=False, ex=None):
            if nx and k in self.kv:
                return False
            self.kv[k] = v.encode()
            return True

        def get(self, k):
            return self.kv.get(k)

        def delete(self, k):
            self.kv.pop(k, None)

    r = R()
    assert fal_ticker.tick_once(r, "me") and calls == ["r", "w"]
    r.kv[fal_ticker.LEASE_KEY] = b"other"
    assert not fal_ticker.tick_once(r, "me")


# ---------------------------------------------------------------------------
# §1 gate round 1 (GPT-6.1 Terra) regressions
# ---------------------------------------------------------------------------


def test_gate1_output_sec_must_match_the_input_duration(env, monkeypatch):
    monkeypatch.setattr(pp, "_probe_duration", lambda path: 300.0)
    with pytest.raises(pp.AdmissionRefused, match="does not match"):
        _admit(env)  # output_sec=5 against a 300 s input


def test_gate1_fetch_with_a_rejected_or_missing_key_keeps_its_attempts(env, monkeypatch):
    tid = _fetching(env)
    env.client.result_outcome = ResultOutcome("retry", 401, error="bad key")
    for _ in range(settings.FAL_FETCH_MAX_ATTEMPTS + 2):
        pp.fal_fetch(tid)
    t = _task(env, tid)
    assert (t.status, t.fetch_attempts, t.error_code) == ("fetching", 0, "auth_rejected")
    assert t.fetch_deadline_at is None and t.claim_owner is None
    monkeypatch.setattr(settings, "FAL_API_KEY", "")
    pp.fal_fetch(tid)
    assert _task(env, tid).fetch_attempts == 0
    monkeypatch.setattr(settings, "FAL_API_KEY", "test-key")
    env.client.result_outcome = ResultOutcome("ok", 200, body={"video": {"url": "https://v3.fal.media/o.mp4"}})
    pp.fal_fetch(tid)
    assert _task(env, tid).status == "done"


def test_gate1_no_deadline_cancel_during_an_auth_hold(env):
    tid = _submitted(env)
    _set(env, tid, error_code="auth_rejected",
         accepted_at=_utcnow() - timedelta(seconds=settings.FAL_POLL_DEADLINE_SEC + 1))
    env.client.status_outcomes = [StatusOutcome("auth", 401)]
    pp.fal_poll(tid)
    assert _task(env, tid).status == "submitted"
    assert not any(fn == "fal_cancel" for fn, _, _ in env.enqueued)
    env.client.status_outcomes = [StatusOutcome("ok", 200, "IN_PROGRESS")]
    pp.fal_poll(tid)  # the 2xx clears the hold ...
    pp.fal_poll(tid)  # ... and the deadline applies again
    assert _task(env, tid).status == "cancelling"


def test_gate2_reservation_is_priced_on_the_longer_of_output_sec_and_the_input(env, monkeypatch):
    monkeypatch.setattr(pp, "_probe_duration", lambda path: 2.5)
    with env.sessions() as s:
        task, _ = pp.admit(
            s, endpoint_id=ENDPOINT, params={}, inputs=[pp.TaskInput("video_url", env.src, "video/mp4")],
            output_sec=1.0, resolution="768p", owner_kind="k", owner_id="g2", owner_project_id=env.pid,
            destination_path=os.path.join(env.proj_dir, "reframe", "g2.mp4"), operation_key="k:g2:main:1")
        assert task.estimated_usd == pytest.approx(3 * 0.08 * 2.0)


def test_gate2_missing_key_restarts_a_running_fetch_window(env, monkeypatch):
    tid = _fetching(env)
    env.client.result_outcome = ResultOutcome("retry", 503, error="busy")
    pp.fal_fetch(tid)  # attempt 1 fails: the window is running
    assert _task(env, tid).fetch_deadline_at is not None
    monkeypatch.setattr(settings, "FAL_API_KEY", "")
    pp.fal_fetch(tid)
    assert _task(env, tid).fetch_deadline_at is None
    monkeypatch.setattr(settings, "FAL_API_KEY", "test-key")
    later = _utcnow() + timedelta(seconds=settings.FAL_FETCH_WINDOW_SEC * 3)
    monkeypatch.setattr(pp, "_now", lambda: later)
    env.client.result_outcome = ResultOutcome("ok", 200, body={"video": {"url": "https://v3.fal.media/o.mp4"}})
    pp.fal_fetch(tid)
    assert _task(env, tid).status == "done"


def test_gate2_a_newly_rejected_key_at_the_deadline_sends_no_cancel(env):
    tid = _submitted(env)
    _set(env, tid, accepted_at=_utcnow() - timedelta(seconds=settings.FAL_POLL_DEADLINE_SEC + 1))
    env.client.status_outcomes = [StatusOutcome("auth", 403)]
    pp.fal_poll(tid)
    t = _task(env, tid)
    assert (t.status, t.error_code) == ("submitted", "auth_rejected")
    assert not any(fn == "fal_cancel" for fn, _, _ in env.enqueued)
    env.client.status_outcomes = [StatusOutcome("transient", 503)]
    pp.fal_poll(tid)  # still held: a transient answer does not lift the hold
    assert _task(env, tid).status == "submitted"


def test_gate3_deadline_cancel_cas_respects_a_hold_recorded_concurrently(env, monkeypatch):
    tid = _submitted(env)
    _set(env, tid, accepted_at=_utcnow() - timedelta(seconds=settings.FAL_POLL_DEADLINE_SEC + 1))
    real_status = env.client.status

    def status_then_concurrent_hold(url):
        out = real_status(url)
        _set(env, tid, error_code="auth_rejected")  # another poll got a 401 meanwhile
        return StatusOutcome("transient", 503)

    monkeypatch.setattr(env.client, "status", status_then_concurrent_hold)
    pp.fal_poll(tid)
    assert _task(env, tid).status == "submitted"
    assert not any(fn == "fal_cancel" for fn, _, _ in env.enqueued)


def test_gate4_a_stale_ok_poll_neither_clears_a_newer_hold_nor_cancels(env, monkeypatch):
    tid = _submitted(env)
    _set(env, tid, error_code="auth_rejected",
         accepted_at=_utcnow() - timedelta(seconds=settings.FAL_POLL_DEADLINE_SEC + 1))
    real_status = env.client.status

    def ok_but_a_newer_hold_lands(url):
        real_status(url)
        # a concurrent poll records a NEW 401 (bumping the version) meanwhile
        with env.sessions() as s:
            s.execute(text("UPDATE provider_tasks SET error_code='auth_rejected', "
                           "version_id=version_id+1 WHERE id=:i"), {"i": tid})
        return StatusOutcome("ok", 200, "IN_PROGRESS")

    monkeypatch.setattr(env.client, "status", ok_but_a_newer_hold_lands)
    pp.fal_poll(tid)
    t = _task(env, tid)
    assert (t.status, t.error_code) == ("submitted", "auth_rejected")
    assert not any(fn == "fal_cancel" for fn, _, _ in env.enqueued)


def test_admission_refuses_without_a_live_fal_worker(env, monkeypatch):
    monkeypatch.setattr(pp, "_fal_worker_alive", lambda: False)
    before = _charged(env)
    with pytest.raises(pp.AdmissionRefused) as exc:
        _admit(env)
    assert exc.value.status_code == 503 and _charged(env) == before


def test_gate6_an_idle_fal_worker_with_a_seven_minute_old_heartbeat_is_alive(monkeypatch):
    import rq

    class W:
        def __init__(self, age, queues=("fal",)):
            self.last_heartbeat = datetime.utcnow() - timedelta(seconds=age)
            self._q = list(queues)

        def queue_names(self):
            return self._q

    monkeypatch.setattr(rq.Worker, "all", classmethod(lambda cls, connection=None: [W(410)]))
    assert pp._live_fal_workers(None)
    monkeypatch.setattr(rq.Worker, "all", classmethod(lambda cls, connection=None: [W(600), W(5, ("process",))]))
    assert not pp._live_fal_workers(None)


def test_q1_result_meta_is_stored_without_any_url(env):
    tid = _fetching(env)
    env.client.result_outcome = ResultOutcome("ok", 200, body={
        "video": {"url": "https://v3.fal.media/out.mp4", "content_type": "video/mp4",
                  "file_size": 123, "width": 432, "height": 768},
        "placement": {"x": 0, "y": 120, "width": 432, "height": 243},
        "seed": 42, "frame_count": 120, "frames_per_second": 24,
        "debug_url": "https://fal.ai/secret", "timings": {"inference": 12.5},
    })
    pp.fal_fetch(tid)
    t = _task(env, tid)
    assert t.status == "done"
    assert t.result_meta == {
        "placement": {"x": 0, "y": 120, "width": 432, "height": 243}, "seed": 42,
        "frame_count": 120, "frames_per_second": 24, "timings": {"inference": 12.5},
        "video": {"content_type": "video/mp4", "file_size": 123, "width": 432, "height": 768},
    }
    assert "://" not in str(t.result_meta)


def test_knowledge_importer_counts_fal_draws_as_unattributable_generations():
    from types import SimpleNamespace

    from app.knowledge.importer import sources
    from app.knowledge.importer.context import Report

    ctx = SimpleNamespace(report=Report(), cursor_tally={}, attempt_for_call_key=lambda key: None)
    draw = SimpleNamespace(id="d-fal", call_key="fal:task:t1", kind="fal_reframe", provider="fal",
                           mcredits=160000, reservation_id="r1")
    sources._spend_draws(ctx, draw)
    rec = ctx.report.extra["reconcile"]
    assert rec["unattributable_draws"] == 1 and rec["unattributable_draws_mcredits"] == 160000
    assert "llm_draws_excluded" not in rec and ctx.cursor_tally == {"unattributable:d-fal": 160000}
