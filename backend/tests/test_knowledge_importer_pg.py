"""Knowledge importer on PostgreSQL (AICP-297; design docs/knowledge-feedback-design.md §6.3, §11).

Opt-in like test_knowledge_pg.py: RESKIN_TEST_PG_URL must name a DISPOSABLE
server. Every test gets its own database, because the importer scans whole
source tables. Source rows are synthetic, created through the ORM; no media
exists, so hashing is injected where a test needs it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import textwrap
import threading
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)")

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_kb_imp_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MANIFEST = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "knowledge", "holdout-manifest-v1.json")
NO_SEED = os.path.join(tempfile.mkdtemp(prefix="reskin_kb_noseed_"))



@pytest.fixture(autouse=True)
def _stub_file_identity(monkeypatch):
    """Importer tests use fake paths: every file has the same stub identity,
    so C2's live check behaves as "still the live final" (real identities are
    covered by test_knowledge_pg.py)."""
    from app.knowledge import store

    monkeypatch.setattr(store, "file_identity", lambda path: "test-live")
    monkeypatch.setattr(store, "_live_identity", lambda session, run_id: "test-live")
    yield

@pytest.fixture()
def factory():
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import sessionmaker

    import app.models  # noqa: F401
    from app.db import Base

    name = "kbi_" + uuid.uuid4().hex[:12]
    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    eng = create_engine(make_url(_PG_URL).set(database=name).render_as_string(hide_password=False), pool_size=10)
    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
    admin.dispose()


def manifest_project(split: str, index: int = -1) -> str:
    body = json.load(open(MANIFEST))
    return [g for g in body["groups"] if g["split"] == split][index]["project_ids"][0]


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run_import(factory, **kw):
    from app.knowledge.importer import run_import as _run

    from app.knowledge.importer.sources import COMMIT_LAG

    kw.setdefault("apply", True)
    kw.setdefault("seed_dir", False)  # seeding off unless a test passes a directory
    # Mutable sources are scanned by updated_at (written "now" by fixtures and by
    # in-test updates): put the clock just past the commit horizon of those writes.
    kw.setdefault("now", datetime.now(timezone.utc) + COMMIT_LAG + timedelta(minutes=1))
    return _run(factory, **kw)


def build_world(s, *, pid=None, project_type="face_swap"):
    """One delivered Run with segments, draws, a promoted audio patch and derived creatives."""
    from app.models import (
        KieSpendDraw, KieSpendReservation, Run, RunAudioPatch, RunCaption, RunSegment, RunTailEdit, VideoProject,
    )

    pid = pid or str(uuid.uuid4())
    if s.get(VideoProject, pid) is None:
        s.add(VideoProject(id=pid, source_type="upload", source_ref="fixture", project_type=project_type))
        s.flush()
    t0 = datetime(2026, 10, 20, 12, 0, 0) + timedelta(seconds=uuid.uuid4().int % 1000)
    run = Run(project_id=pid, model="seedance-2-5", status="done", language="es",
              result_local_path=f"/data/{pid}/patched.mp4", result_gdrive_file_id="drive-" + pid[:8],
              final_creatives_exported_at=t0 + timedelta(hours=5), created_at=t0,
              segment_snapshot=[{"id": "sd-a", "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 5.0},
                                {"id": "sd-b", "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 9.5}])
    s.add(run)
    s.flush()
    t1, t2, t3, old = (f"task-{run.id[:8]}-{n}" for n in ("1", "2", "3", "old"))
    s.add_all([
        RunSegment(run_id=run.id, segment_def_id="sd-a", index=0, status="completed", seedance_task_id=t1,
                   cost_credits=12.5, local_result_path=f"/data/{run.id}/result_0000.mp4",
                   created_at=t0 + timedelta(minutes=1)),
        RunSegment(run_id=run.id, segment_def_id="sd-b", index=1, status="failed", seedance_task_id=t2,
                   error_message="kie: internal error", created_at=t0 + timedelta(minutes=2)),
        RunSegment(run_id=run.id, segment_def_id="sd-c", index=2, status="completed", seedance_task_id=t3,
                   cost_credits=7.0, local_result_path=f"/data/{run.id}/result_0002.mp4",
                   created_at=t0 + timedelta(minutes=3)),
    ])
    res = KieSpendReservation(scope_key=f"run:{run.id}", day=date(2026, 10, 20), kind="new run", run_id=run.id)
    s.add(res)
    s.flush()
    s.add_all([
        KieSpendDraw(call_key=t1, reservation_id=res.id, kind="video", estimated_mcredits=10000, mcredits=12000,
                     created_at=t0 + timedelta(minutes=1), settled_at=t0 + timedelta(minutes=9)),
        KieSpendDraw(call_key=old, reservation_id=res.id, kind="video", estimated_mcredits=9000, mcredits=9000,
                     created_at=t0 + timedelta(seconds=30)),
    ])
    llm = KieSpendReservation(scope_key=f"localise:{pid}", day=date(2026, 10, 20), kind="translate")
    s.add(llm)
    s.flush()
    s.add(KieSpendDraw(call_key=f"translate:{uuid.uuid4().hex}", reservation_id=llm.id, kind="translate",
                       estimated_mcredits=500, mcredits=500, created_at=t0 + timedelta(seconds=10)))
    s.add(RunAudioPatch(run_id=run.id, donor_run_id=str(uuid.uuid4()), target_start_sec=1.0, target_end_sec=2.0,
                        donor_start_sec=1.0, crossfade_sec=0.02, base_result_path=f"/data/{pid}/final.mp4",
                        status="done", output_path=f"/data/{pid}/patched.mp4", promoted_at=t0 + timedelta(hours=2),
                        created_at=t0 + timedelta(hours=1)))
    s.add(RunTailEdit(run_id=run.id, cut_sec=8.0, source_type="run", status="done",
                      output_path=f"/data/{run.id}/tail.mp4", final_creatives_exported_at=t0 + timedelta(hours=6),
                      created_at=t0 + timedelta(hours=3)))
    s.add(RunCaption(run_id=run.id, source_type="run", status="failed", error_message="gemini down",
                     created_at=t0 + timedelta(hours=4)))
    s.commit()
    return pid, run.id, (t1, t2, t3, old)


def kb_counts(s) -> dict:
    from sqlalchemy import func, select

    from app.knowledge.models import ALL_TABLES
    from app.db import Base

    out = {}
    for name in ALL_TABLES:
        out[name] = s.execute(select(func.count()).select_from(Base.metadata.tables[name])).scalar()
    return out


def evidence_counts(s) -> dict:
    skip = {"kb_import_batches", "kb_import_batch_observations", "kb_import_cursors"}
    return {k: v for k, v in kb_counts(s).items() if k not in skip}


def attempt_by_task(s, task_id):
    from sqlalchemy import select

    from app.knowledge.models import KbAttempt, KbAttemptSubmission

    sub = s.execute(select(KbAttemptSubmission).where(KbAttemptSubmission.provider_task_id == task_id)).scalar_one()
    return s.get(KbAttempt, sub.attempt_id)


# ---------------------------------------------------------------- tests


def test_apply_maps_sources_and_rerun_is_idempotent(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbArtifact, KbAttempt, KbCase, KbFeedbackEvent

    s = factory()
    pid, rid, (t1, t2, t3, old) = build_world(s)
    report = run_import(factory)
    assert report["outcome"] == "finished" and report["failures"]["total"] == 0, report["failures"]

    run_case = store.current_case(s, store.run_case_key(rid))
    assert run_case.origin == "import" and run_case.import_batch_id == report["batch_id"]
    vs = store.versions(s, run_case.id)
    assert [(v.reason, s.get(KbArtifact, v.artifact_id).locator) for v in vs] == [
        ("import", f"/data/{pid}/final.mp4"), ("audio_patch", f"/data/{pid}/patched.mp4")]
    assert all(s.get(KbArtifact, v.artifact_id).sha256 is None for v in vs)  # hashing off: unknown stays NULL

    events = s.execute(select(KbFeedbackEvent)).scalars().all()
    assert not [e for e in events if e.event_type == "operator_verdict"]
    exports = [e for e in events if e.event_type == "export_observed"]
    assert len(exports) == 2 and all(e.actor_kind == "importer" and e.verdict is None for e in exports)
    correction = next(e for e in events if e.event_type == "correction")
    assert (correction.before_artifact_id, correction.after_artifact_id) == (vs[0].artifact_id, vs[1].artifact_id)
    assert correction.resolved == "unknown" and correction.method == "automatic"
    assert store.case_state(s, run_case.id)["verdict"] == "unreviewed"

    # Settled vs estimated: draw cost on the attempt that has a draw, labeled snapshot cost on the one without.
    a1 = store.attempt_state(s, attempt_by_task(s, t1).id)
    assert (a1["status"], a1["estimated_mcredits"], a1["settled_mcredits"], a1["cost_source"]) == (
        "succeeded", 10000, 12000, "spend_draw")
    a2 = store.attempt_state(s, attempt_by_task(s, t2).id)
    assert a2["status"] == "failed" and a2["settled_mcredits"] is None and a2["estimated_mcredits"] is None
    a3_obs = store.attempt_state(s, attempt_by_task(s, t3).id)["observations"]
    snap = [o for o in a3_obs if o.kind == "cost_settled"]
    assert [(o.mcredits, o.cost_source, o.payload["label"]) for o in snap] == [
        (7000, "run_segment_snapshot", "current attempt at import time")]
    # The superseded paid call becomes an ambiguous attempt with the draw's estimate only.
    amb = store.attempt_state(s, attempt_by_task(s, old).id)
    assert any(o.kind == "ambiguous" for o in amb["observations"])
    assert (amb["estimated_mcredits"], amb["settled_mcredits"]) == (9000, None)
    assert amb["provider_task_id"] == old and len([a for a in s.execute(select(KbAttempt)).scalars()
                                                     if a.target_kind == "spend_draw"]) == 1
    assert report["reconcile"]["llm_draws_excluded"] == 1  # LLM calls are never attempts (R9)
    seg = store.current_case(s, store.segment_case_key(rid, "sd-a"))
    assert (seg.parent_case_id, seg.time_start_sec, seg.time_end_sec) == (run_case.id, 0.0, 5.0)
    assert "time_range" in store.current_case(s, store.segment_case_key(rid, "sd-c")).missing_facts
    derived = s.execute(select(KbCase).where(KbCase.scope == "derived")).scalars().all()
    assert sorted(c.case_key.split(":")[1] for c in derived) == ["run_captions", "run_tail_edits"]
    assert all(c.parent_case_id == run_case.id and c.origin == "import" for c in derived)

    before = evidence_counts(s)
    again = run_import(factory)  # cursors: nothing new
    assert evidence_counts(s) == before and again["created"] == {}
    full = run_import(factory, from_start=True)  # full rescan: still nothing twice
    assert full["failures"]["total"] == 0, full["failures"]
    assert evidence_counts(s) == before and full["created"] == {}
    assert len({a.idempotency_key for a in s.execute(select(KbAttempt)).scalars()}) == before["kb_attempts"]


def test_dry_run_writes_nothing_and_reports_everything(factory):
    s = factory()
    build_world(s)
    report = run_import(factory, apply=False)
    assert report["dry_run"] is True and report["outcome"] == "finished"
    assert report["created"]["case"] >= 5 and report["created"]["attempt"] >= 4
    assert report["missing_media"]["total"] > 0  # no file exists in the test
    assert all(n == 0 for n in kb_counts(factory()).values())


def test_incremental_cursor_and_bounded_ticks(factory):
    from app.models import RunTailEdit

    s = factory()
    _, rid, _ = build_world(s)
    first = run_import(factory, limit=2)
    assert sum(v["rows"] for v in first["sources"].values()) == 2  # bounded, whichever sources it rotated to
    while True:
        r = run_import(factory, limit=2)
        if not any(v.get("rows") for v in r["sources"].values()):
            break
    complete = evidence_counts(s)
    s.add(RunTailEdit(run_id=rid, cut_sec=3.0, source_type="run", status="done", output_path="/data/t2.mp4",
                      created_at=datetime(2026, 10, 26)))
    s.commit()
    nxt = run_import(factory)
    assert nxt["sources"]["tail_edits"]["rows"] == 1 and nxt["sources"]["runs"]["rows"] == 0
    after = evidence_counts(s)
    assert after["kb_cases"] == complete["kb_cases"] + 1


def test_malformed_and_conflicting_rows_are_reported_not_fatal(factory):
    from app.knowledge import store
    from app.models import Run, RunSegment, VideoProject

    s = factory()
    pid, rid, (t1, *_) = build_world(s)
    # A live case already bound to a different identity → conflict for this run only.
    other_pid = str(uuid.uuid4())
    s.add(VideoProject(id=other_pid, source_type="upload", source_ref="x", project_type="face_swap"))
    bad = Run(project_id=other_pid, status="done", result_local_path="/data/bad.mp4",
              created_at=datetime(2026, 10, 25))
    s.add(bad)
    s.flush()
    store.ensure_case(s, case_key=store.run_case_key(bad.id), scope="run", operation="cover_change",
                      project_id=other_pid, run_id=bad.id)
    # The same provider task on two runs → the second is a conflict, recorded without a submission.
    dup = Run(project_id=pid, status="done", created_at=datetime(2026, 10, 25))
    s.add(dup)
    s.flush()
    s.add(RunSegment(run_id=dup.id, segment_def_id="sd-x", index=0, status="completed", seedance_task_id=t1,
                     created_at=datetime(2026, 10, 25)))
    s.commit()
    report = run_import(factory)
    assert report["outcome"] == "finished"
    assert report["conflicts"]["total"] >= 2
    texts = json.dumps(report["conflicts"]["items"])
    assert bad.id in texts and "already attributed" in texts
    assert store.current_case(s, store.run_case_key(rid)) is not None  # the rest still imported
    assert report["missing_metadata"]["total"] >= 1  # sd-c / sd-x are not in any segment_snapshot


def _seed_file(tmp_path, body: str) -> str:
    d = tmp_path / "seed"
    d.mkdir(exist_ok=True)
    (d / "fixture.yaml").write_text(textwrap.dedent(body))
    return str(d)


def test_holdout_membership_and_refused_holdout_evidence(factory, tmp_path):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbEvalMembership, KbRuleEvidence, KbRuleVersion

    s = factory()
    hold_pid, dev_pid = manifest_project("holdout"), manifest_project("development")
    _, hold_rid, _ = build_world(s, pid=hold_pid)
    _, dev_rid, _ = build_world(s, pid=dev_pid)
    _, out_rid, _ = build_world(s)
    seed_dir = _seed_file(tmp_path, f"""
        rules:
          - slug: o1-gemini-omni-text-replacement
            kind: operator_constraint
            statement: "Avoid Gemini Omni for text replacement."
            applicability: {{operation: [subtitle_removal]}}
            recommendation: {{action: warn, target: {{model_key: gemini-omni}}}}
            confidence: operator_asserted
            shareable: false
            effective_from: "2026-10-26T00:00:00Z"
            author_label: "Operator (AICP-293)"
            provenance: {{source: "operator", linear: "AICP-293"}}
            evidence_gaps: ["failing runs unrecorded"]
            evidence:
              - {{run_id: "{dev_rid}", stance: supports, note: dev}}
              - {{run_id: "{uuid.uuid4()}", stance: context}}
          - slug: holdout-cited
            kind: empirical_guidance
            statement: "Cites a holdout run."
            recommendation: {{action: prefer, target: {{model_key: seedance}}}}
            confidence: low
            effective_from: "2026-10-26T00:00:00Z"
            evidence:
              - {{run_id: "{hold_rid}", stance: supports}}
          - slug: broken
            kind: empirical_guidance
            statement: "missing confidence"
            recommendation: {{action: warn, target: {{model_key: seedance}}}}
            effective_from: "2026-10-26T00:00:00Z"
    """)
    report = run_import(factory, seed_dir=seed_dir)
    members = {m.case_id: m for m in s.execute(select(KbEvalMembership)).scalars()}
    hold_case = store.current_case(s, store.run_case_key(hold_rid))
    dev_case = store.current_case(s, store.run_case_key(dev_rid))
    assert members[hold_case.id].split == "holdout" and members[dev_case.id].split == "development"
    out_case = store.current_case(s, store.run_case_key(out_rid))
    assert out_case.id not in members
    assert any(i["case_key"] == out_case.case_key for i in report["excluded_cases"]["items"])
    assert report["eval"]["assigned"] == len(members)

    versions = s.execute(select(KbRuleVersion)).scalars().all()
    assert len(versions) == 1 and versions[0].author_kind == "importer"
    assert versions[0].author_label == "Operator (AICP-293)" and versions[0].import_batch_id == report["batch_id"]
    assert [e.case_id for e in s.execute(select(KbRuleEvidence)).scalars()] == [dev_case.id]
    assert [i["slug"] for i in report["refused"]["items"]] == ["holdout-cited"]
    assert "holdout" in report["refused"]["items"][0]["error"]
    assert any("no current case" in i["what"] for i in report["skipped"]["items"])
    assert [i.get("slug") for i in report["failures"]["items"]] == ["broken"]
    # Unchanged YAML: no new version; membership stays fixed by the manifest.
    again = run_import(factory, seed_dir=seed_dir, from_start=True)
    assert len(s.execute(select(KbRuleVersion)).scalars().all()) == 1
    assert any(i.get("slug") == "o1-gemini-omni-text-replacement" for i in again["duplicates"]["items"])
    assert {m.case_id: m.split for m in s.execute(select(KbEvalMembership)).scalars()} == {
        k: v.split for k, v in members.items()}


def test_invalidated_batch_reimports_as_revision_two(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt, KbAttemptObservation

    s = factory()
    _, rid, (t1, _, _, old) = build_world(s)
    first = run_import(factory)
    store.invalidate(s, target_kind="import_batch", target_id=first["batch_id"], reason="wrong mapping")
    s.commit()
    from app.knowledge.access import KnowledgeScope
    from app.assistant_access import ProjectAllowlist

    scope = KnowledgeScope.build(ProjectAllowlist(all_projects=True), "*")
    assert store.run_timeline(s, scope, rid)["cases"] == []
    second = run_import(factory, from_start=True)
    assert second["failures"]["total"] == 0, second["failures"]
    case = store.current_case(s, store.run_case_key(rid))
    assert case.revision == 2 and case.import_batch_id == second["batch_id"]
    view = store.run_timeline(s, scope, rid)["cases"]
    assert view[0]["id"] == case.id and view[0]["state"]["latest_version_no"] == 2
    invalid = store.InvalidSet.load(s)
    valid = [a for a in s.execute(select(KbAttempt)).scalars() if not invalid.attempt_invalid(a)]
    assert len(valid) == len(s.execute(select(KbAttempt)).scalars().all()) // 2
    # Task ids stay unique across withdrawn submissions: the replacement carries them in its status payload,
    # and the draw matcher still finds it (no second ambiguous attempt, costs attached).
    withheld = [o for o in s.execute(select(KbAttemptObservation)).scalars()
                if o.payload and o.payload.get("submission_withheld") and o.payload["provider_task_id"] == t1]
    assert len(withheld) == 1
    state = store.attempt_state(s, withheld[0].attempt_id)
    assert (state["estimated_mcredits"], state["settled_mcredits"]) == (10000, 12000)
    amb = [a for a in valid if any(o.kind == "ambiguous" for o in store.attempt_state(s, a.id)["observations"])]
    assert len(amb) == 1


def test_stale_intents_and_changed_finals_are_reconciled(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbArtifact

    s = factory()
    pid, rid, _ = build_world(s)
    content = {f"/data/{pid}/patched.mp4": sha("v1")}
    hasher = content.__getitem__
    run_import(factory, hasher=hasher, exists=lambda p: p in content)
    case = store.current_case(s, store.run_case_key(rid))
    latest = store.latest_version(s, case.id)
    assert s.get(KbArtifact, latest.artifact_id).sha256 == sha("v1")
    intent, _ = store.record_attempt_intent(s, case_id=case.id, attempt_kind="generation", origin="automatic",
                                            provider="kie", request={"prompt": "p"}, target_kind="run_segment",
                                            target_id="rs-1")
    s.commit()
    content[f"/data/{pid}/patched.mp4"] = sha("v2")  # restitched in place
    later = datetime.now(timezone.utc) + timedelta(days=1)
    report = run_import(factory, hasher=hasher, exists=lambda p: p in content, now=later)
    assert report["reconcile"]["finals_queued"] == 1 and report["reconcile"]["capture_appended"] == 1
    assert report["reconcile"]["intents_without_submission"] == 1
    assert store.attempt_state(s, intent.id)["status"] == "intent_without_submission"
    vs = store.versions(s, case.id)
    assert vs[-1].source_record == f"reconcile:{rid}:{sha('v2')}:{latest.version_no}"
    again = run_import(factory, hasher=hasher, exists=lambda p: p in content, now=later, from_start=True)
    assert again["reconcile"].get("finals_queued", 0) == 0 and len(store.versions(s, case.id)) == len(vs)
    assert not s.execute(select(KbArtifact).where(KbArtifact.sha256 == sha("v2"))).scalars().all()[1:]


class FakeRedis:
    def __init__(self):
        self.data, self.lock = {}, threading.Lock()

    def set(self, key, value, nx=False, ex=None):
        with self.lock:
            if nx and key in self.data:
                return None
            self.data[key] = value
            return True

    def eval(self, script, numkeys, key, token, *args):
        with self.lock:
            if self.data.get(key) != token:
                return 0
            if "EXPIRE" not in script:
                del self.data[key]
            return 1


def test_leases_keep_concurrent_ticks_apart(factory, monkeypatch):
    from app.knowledge.importer import run_tick
    from app.knowledge.importer.runner import REDIS_LEASE_KEY, acquire_lease, release_lease

    s = factory()
    build_world(s)
    redis = FakeRedis()
    redis.set(REDIS_LEASE_KEY, "someone-else")
    assert "outcome" not in run_tick(factory, redis, hasher=None)
    redis.data.clear()

    # A cursor lease held by another importer keeps this one off that source.
    assert acquire_lease(factory, "runs", "other-owner")
    held = run_import(factory, sources=["runs", "run_segments"])
    assert held["leased_elsewhere"] == ["runs"] and "runs" not in held["sources"]
    assert not acquire_lease(factory, "runs", "third-owner")
    release_lease(factory, "runs", "other-owner")

    results, barrier = [], threading.Barrier(2)

    def tick():
        barrier.wait()
        results.append(run_tick(factory, redis, hasher=None))

    threads = [threading.Thread(target=tick) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(r.get("outcome") == "finished" for r in results) == [False, True]
    assert any(r.get("skipped") == "another knowledge import tick holds the lease" for r in results)
    assert REDIS_LEASE_KEY not in redis.data  # released

    from app import tasks
    monkeypatch.setattr(tasks.settings, "KNOWLEDGE_CAPTURE_ENABLED", False)
    assert tasks.knowledge_import_tick() is None


def test_tick_import_is_opt_in_even_with_capture_on(monkeypatch):
    """The importer reads every project, so capture alone runs only the C2 drain (AICP-298)."""
    from app import tasks
    from app.knowledge import importer

    called = []
    monkeypatch.setattr(importer, "run_tick", lambda *a, **k: called.append(k["drain_only"]) or {"outcome": "finished"})
    monkeypatch.setattr(tasks, "_get_redis", lambda: None)
    monkeypatch.setattr(tasks.settings, "KNOWLEDGE_CAPTURE_ENABLED", True)
    monkeypatch.setattr(tasks.settings, "KNOWLEDGE_IMPORT_TICK", False, raising=False)
    assert tasks.knowledge_import_tick() == {"outcome": "finished"} and called == [True]
    monkeypatch.setattr(tasks.settings, "KNOWLEDGE_IMPORT_TICK", True)
    assert tasks.knowledge_import_tick() == {"outcome": "finished"} and called == [True, False]


def test_drain_only_tick_reads_no_historical_source(factory):
    from sqlalchemy import select

    from app.knowledge.importer import run_tick
    from app.knowledge.models import KbCase, KbImportCursor

    s = factory()
    build_world(s)
    s.close()
    report = run_tick(factory, FakeRedis(), hasher=None, drain_only=True)
    assert report["outcome"] == "finished", report
    assert not report["sources"], report["sources"]  # no runs/segments/draws pass
    s = factory()
    assert s.execute(select(KbCase)).first() is None  # nothing imported from history
    names = {c.source for c in s.execute(select(KbImportCursor)).scalars()}
    assert names <= {"reconcile"}
    s.close()


# ---------------------------------------------------------------- Terra review regressions (AICP-297 r1)


def test_rows_inside_the_commit_horizon_wait_and_late_commits_are_not_skipped(factory):
    from app.knowledge.importer.sources import COMMIT_LAG
    from app.models import RunTailEdit

    s = factory()
    _, rid, _ = build_world(s)  # tail edit at ~15:00, caption at ~16:00 on 2026-10-20
    from sqlalchemy import text as _text
    for table in ("runs", "run_segments"):  # fixture rows were written then and not touched since
        s.execute(_text(f"UPDATE {table} SET updated_at = created_at"))
    s.commit()
    base = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    first = run_import(factory, now=base + COMMIT_LAG + timedelta(minutes=40))
    assert first["sources"]["runs"]["rows"] == 1 and first["sources"]["tail_edits"]["rows"] == 0
    # A row whose transaction committed late, with a created_at before the newer tail edit.
    s.add(RunTailEdit(run_id=rid, cut_sec=2.0, source_type="run", status="done", output_path="/data/late.mp4",
                      created_at=datetime(2026, 10, 20, 14, 0)))
    s.commit()
    second = run_import(factory, now=base + COMMIT_LAG + timedelta(hours=5))
    assert second["sources"]["tail_edits"]["rows"] == 2


def test_non_terminal_rows_are_revisited_until_terminal(factory):
    from app.knowledge import store
    from app.models import Run, RunSegment, VideoProject

    s = factory()
    pid = str(uuid.uuid4())
    s.add(VideoProject(id=pid, source_type="upload", source_ref="x", project_type="face_swap"))
    run = Run(project_id=pid, status="processing", created_at=datetime(2026, 10, 19),
              segment_snapshot=[{"id": "sd-a", "start_sec": 0.0, "end_sec": 4.0}])
    s.add(run)
    s.flush()
    seg = RunSegment(run_id=run.id, segment_def_id="sd-a", index=0, status="generating", seedance_task_id="tk-live",
                     created_at=datetime(2026, 10, 19, 0, 1))
    s.add(seg)
    s.commit()
    first = run_import(factory)
    assert first["sources"]["runs"]["pending"] == 1 and first["sources"]["run_segments"]["pending"] == 1
    case = store.current_case(s, store.run_case_key(run.id))
    assert store.versions(s, case.id) == []
    run.status, run.result_local_path = "done", "/data/live/final.mp4"
    seg.status, seg.local_result_path = "completed", "/data/live/result_0000.mp4"
    s.commit()
    second = run_import(factory)
    assert second["sources"]["runs"]["revisit_done"] == 1 and second["sources"]["runs"]["pending"] == 0
    assert [v.reason for v in store.versions(s, case.id)] == ["import"]
    assert store.attempt_state(s, attempt_by_task(s, "tk-live").id)["status"] == "succeeded"
    assert run_import(factory)["created"] == {}  # nothing pending any more


def test_final_reconciliation_handles_a_case_without_versions(factory):
    from app.knowledge import store
    from app.models import Run, VideoProject

    s = factory()
    pid = str(uuid.uuid4())
    s.add(VideoProject(id=pid, source_type="upload", source_ref="x", project_type="face_swap"))
    run = Run(project_id=pid, status="done", result_local_path="/data/nov/final.mp4")
    s.add(run)
    s.flush()
    case = store.ensure_case(s, case_key=store.run_case_key(run.id), scope="run", operation="face_swap",
                             project_id=pid, run_id=run.id)
    s.commit()
    report = run_import(factory, sources=["reconcile"], hasher=lambda p: sha(p), exists=lambda p: True)
    assert report["reconcile"]["finals_queued"] == 1
    vs = store.versions(s, case.id)
    assert [v.source_record for v in vs] == [f"reconcile:{run.id}:{sha('/data/nov/final.mp4')}:0"]


def test_lost_leases_stop_the_pass_before_its_next_commit(factory):
    from sqlalchemy import text

    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    _, rid, _ = build_world(s)
    calls = []

    def flaky() -> bool:  # the Redis lease lapses after the first row
        calls.append(1)
        return len(calls) == 1

    report = run_import(factory, keepalive=flaky)
    assert report["outcome"] == "aborted" and "LeaseLost" in json.dumps(report["failures"])
    assert read_cursor(s, "runs")["id"] == rid
    assert (read_cursor(s, "run_segments") or {}).get("ts") is None  # its first row was rolled back
    assert store.current_case(s, store.segment_case_key(rid, "sd-a")) is None

    def thief() -> bool:  # another importer takes the cursor lease between rows
        with factory() as other:
            other.execute(text("UPDATE kb_import_cursors SET lease_owner = 'thief' WHERE lease_owner = 'me'"))
            other.commit()
        return True

    stolen = run_import(factory, owner="me", keepalive=thief)
    assert stolen["outcome"] == "aborted"
    assert store.current_case(s, store.segment_case_key(rid, "sd-a")) is None
    assert s.execute(text("SELECT count(*) FROM kb_import_cursors WHERE lease_owner = 'thief'")).scalar() > 0


def test_default_seed_directory_loads_every_rule_in_a_dry_run(factory):
    from app.knowledge.importer.seed import load_seed_files

    n = len(load_seed_files())  # every rule in the shipped directory, however many are added
    assert n >= 11
    report = run_import(factory, apply=False, sources=["seed"], seed_dir=None)
    assert report["sources"]["seed"] == {"rows": n, "failed": 0}
    assert report["created"]["rule_version"] == n
    assert report["refused"]["total"] == 0 and report["failures"]["total"] == 0
    assert all(n == 0 for n in kb_counts(factory()).values())


def test_export_observed_takes_the_run_lock_first(factory):
    from sqlalchemy import select, text
    from sqlalchemy.exc import OperationalError

    from app.knowledge import store
    from app.models import Run, VideoProject

    s = factory()
    pid = str(uuid.uuid4())
    s.add(VideoProject(id=pid, source_type="upload", source_ref="x", project_type="face_swap"))
    run = Run(project_id=pid, status="done")
    s.add(run)
    s.flush()
    case = store.ensure_case(s, case_key=store.run_case_key(run.id), scope="run", operation="face_swap",
                             project_id=pid, run_id=run.id)
    s.commit()
    holder = factory()
    holder.execute(select(Run.id).where(Run.id == run.id).with_for_update())
    s.execute(text("SET lock_timeout = '300ms'"))
    with pytest.raises(OperationalError):
        store.record_export_observed(s, case_id=case.id, idempotency_key="exp-lock", source_record="runs:x",
                                     exported_at=datetime(2026, 10, 21, tzinfo=timezone.utc))
    s.rollback()
    holder.rollback()
    holder.close()
    receipt = store.record_export_observed(s, case_id=case.id, idempotency_key="exp-lock", source_record="runs:x",
                                           exported_at=datetime(2026, 10, 21, tzinfo=timezone.utc))
    assert len(receipt.ids) == 1


def test_reconcile_scans_are_bounded_per_pass(factory):
    from app.models import KieSpendDraw, KieSpendReservation, Run
    from sqlalchemy import select

    s = factory()
    _, rid, _ = build_world(s)
    res = s.execute(select(KieSpendReservation).where(KieSpendReservation.run_id == rid)).scalar_one()
    extra = [KieSpendDraw(call_key=f"lost-{n}-{rid[:6]}", reservation_id=res.id, kind="video",
                          estimated_mcredits=1000, mcredits=1000, created_at=datetime(2026, 10, 20, 13, n))
             for n in range(3)]
    s.add_all(extra)
    s.commit()
    run_import(factory)
    for n, d in enumerate(extra):
        d.mcredits, d.settled_at = 1500 + n, datetime(2026, 10, 21, 9, n)
    s.commit()
    got = [run_import(factory, sources=["reconcile"], limit=1)["reconcile"]["settlements_recorded"]
           for _ in range(4)]
    assert got == [1, 1, 1, 0]
    assert s.get(Run, rid) is not None


def test_probed_media_writes_availability_checks(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbArtifact, KbArtifactCheck

    s = factory()
    pid, rid, _ = build_world(s)
    current = f"/data/{pid}/patched.mp4"
    run_import(factory, hasher=lambda p: sha(p), exists=lambda p: p == current)
    checks = {s.get(KbArtifact, c.artifact_id).locator: (c.present, c.sha256_matches)
              for c in s.execute(select(KbArtifactCheck)).scalars()}
    assert checks[current] == (True, True)
    assert checks[f"/data/{pid}/final.mp4"] == (False, None)
    assert checks[f"/data/{rid}/result_0000.mp4"] == (False, None)
    before = len(checks)
    run_import(factory, hasher=lambda p: sha(p), exists=lambda p: p == current, from_start=True)
    assert len(s.execute(select(KbArtifactCheck)).scalars().all()) == before


def test_manual_creative_uses_its_copied_provenance_before_the_overlay(factory):
    from app.knowledge import store
    from app.models import ManualCreative

    s = factory()
    pid, rid, _ = build_world(s)
    by_run = ManualCreative(pipeline_type="re-skin-manual-script", idea="x", gdrive_file_id="g-" + rid[:8],
                            overlay_job_id=str(uuid.uuid4()),  # overlay hard-deleted
                            details={"run_id": rid}, created_at=datetime(2026, 10, 20, 18))
    by_project = ManualCreative(pipeline_type="re-skin-manual-script", idea="y", gdrive_file_id="h-" + rid[:8],
                                details={"project_id": pid, "source_type": "upload"},
                                created_at=datetime(2026, 10, 20, 18, 1))
    orphan = ManualCreative(pipeline_type="re-skin-manual-script", idea="z", gdrive_file_id="i-" + rid[:8],
                            details={"source_type": "upload"}, created_at=datetime(2026, 10, 20, 18, 2))
    s.add_all([by_run, by_project, orphan])
    s.commit()
    report = run_import(factory)
    a = store.current_case(s, store.manual_case_key(by_run.id))
    assert (a.run_id, a.parent_case_id) == (rid, store.current_case(s, store.run_case_key(rid)).id)
    assert a.snapshot["attributed_via"] == "details.run_id"
    b = store.current_case(s, store.manual_case_key(by_project.id))
    assert (b.project_id, b.run_id, b.parent_case_id) == (pid, None, None) and "run" in b.missing_facts
    assert store.current_case(s, store.manual_case_key(orphan.id)) is None
    assert any(i.get("manual_creative_id") == orphan.id for i in report["skipped"]["items"])


def test_only_done_runs_contribute_finals(factory):
    from app.knowledge import store
    from app.models import Run, VideoProject

    s = factory()
    pid = str(uuid.uuid4())
    s.add(VideoProject(id=pid, source_type="upload", source_ref="x", project_type="face_swap"))
    failed = Run(project_id=pid, status="failed", result_local_path="/data/f/final.mp4",
                 created_at=datetime(2026, 10, 19))
    done = Run(project_id=pid, status="done", result_local_path="/data/d/final.mp4",
               created_at=datetime(2026, 10, 19, 1))
    s.add_all([failed, done])
    s.commit()
    every = dict(hasher=lambda p: sha(p), exists=lambda p: True)
    run_import(factory, sources=["reconcile"], **every)  # S9a: no case yet → ensured and queued
    case = store.current_case(s, store.run_case_key(done.id))
    assert [v.source_record for v in store.versions(s, case.id)] == [
        f"reconcile:{done.id}:{sha('/data/d/final.mp4')}:0"]
    assert store.current_case(s, store.run_case_key(failed.id)) is None  # S9b: never a failed run's final
    report = run_import(factory, **every)
    fcase = store.current_case(s, store.run_case_key(failed.id))
    assert store.versions(s, fcase.id) == [] and report["sources"]["runs"]["pending"] == 1
    assert len(store.versions(s, case.id)) == 1  # the runs source finds the reconciled version


def test_importer_reuses_the_attempt_that_already_holds_a_task(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt, KbAttemptObservation, KbAttemptSubmission
    from app.models import Run, RunSegment

    s = factory()
    pid, rid, (t1, *_) = build_world(s)
    run = s.get(Run, rid)
    rs = s.execute(select(RunSegment).where(RunSegment.seedance_task_id == t1)).scalar_one()
    root = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid)
    seg = store.ensure_case(s, case_key=store.segment_case_key(rid, "sd-a"), scope="segment", operation="face_swap",
                            project_id=pid, run_id=rid, run_segment_id=rs.id, segment_def_id="sd-a",
                            parent_case_id=root.id)
    orphan, _ = store.record_attempt_intent(s, case_id=seg.id, attempt_kind="generation", origin="automatic",
                                            provider="kie", request=None, idempotency_key=f"orphan:kie:{t1}")
    store.record_submission(s, orphan.id, provider_task_id=t1, spend_call_key=t1)
    s.commit()
    assert run is not None
    report = run_import(factory)
    assert report["failures"]["total"] == 0, report["failures"]
    holders = s.execute(select(KbAttemptSubmission).where(KbAttemptSubmission.provider_task_id == t1)).scalars().all()
    assert [h.attempt_id for h in holders] == [orphan.id]
    withheld = [o for o in s.execute(select(KbAttemptObservation)).scalars()
                if (o.payload or {}).get("provider_task_id") == t1]
    assert withheld == []
    state = store.attempt_state(s, orphan.id)
    assert (state["status"], state["settled_mcredits"]) == ("succeeded", 12000)
    assert not [a for a in s.execute(select(KbAttempt)).scalars()
                if a.id != orphan.id and a.target_id == rs.id and a.attempt_kind == "generation"]


# ---------------------------------------------------------------- Terra review regressions (AICP-297 r2)


def _simple_run(s, status, created_at, **kw):
    from app.models import Run, VideoProject

    pid = str(uuid.uuid4())
    s.add(VideoProject(id=pid, source_type="upload", source_ref="x", project_type="face_swap"))
    kw.setdefault("updated_at", created_at)  # written then, untouched since (sources scan updated_at)
    run = Run(project_id=pid, status=status, created_at=created_at, **kw)
    s.add(run)
    s.commit()
    return run


def test_dry_run_takes_no_run_locks(factory):
    from sqlalchemy import select

    from app.models import Run

    s = factory()
    _, rid, _ = build_world(s)  # its run and tail edit carry Final Creatives stamps
    holder = factory()
    holder.execute(select(Run.id).where(Run.id == rid).with_for_update())
    out = []
    th = threading.Thread(target=lambda: out.append(run_import(factory, apply=False)))
    th.start()
    th.join(90)
    holder.rollback()
    holder.close()
    th.join(30)
    assert out and out[0]["outcome"] == "finished", "dry run waited on a production Run lock"
    assert out[0]["reconcile"]["exports_observed_dry_run"] == 2


def test_export_reconciler_commits_per_stamp_and_skips_a_locked_run(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbFeedbackEvent
    from app.models import Run

    s = factory()
    _, rid, _ = build_world(s)
    run = s.get(Run, rid)
    run.final_creatives_exported_at = None
    s.commit()
    run_import(factory)
    run.final_creatives_exported_at = datetime(2026, 10, 21, 9)
    s.commit()
    holder = factory()
    holder.execute(select(Run.id).where(Run.id == rid).with_for_update())
    blocked = run_import(factory, sources=["reconcile"])
    holder.rollback()
    holder.close()
    assert blocked["reconcile"]["exports_locked_skipped"] == 1
    exports = lambda: [e for e in s.execute(select(KbFeedbackEvent)).scalars()  # noqa: E731
                       if e.event_type == "export_observed" and "source=runs:" in e.note]
    assert exports() == []
    run_import(factory, sources=["reconcile"])
    assert len(exports()) == 1


def test_final_reconciliation_stops_at_the_deadline(factory):
    import time

    from app.knowledge.importer.runner import read_cursor

    s = factory()
    runs = [_simple_run(s, "done", datetime(2026, 10, 19, 0, n), result_local_path=f"/data/r{n}/final.mp4")
            for n in range(3)]
    hashed = []

    def slow(path):
        hashed.append(path)
        time.sleep(1.5)
        return sha(path)

    report = run_import(factory, sources=["reconcile"], hasher=slow, exists=lambda p: True,
                        deadline=datetime.now(timezone.utc) + timedelta(seconds=1))
    assert hashed == [runs[0].result_local_path] and report["reconcile"]["finals_queued"] == 1
    assert read_cursor(s, "reconcile")["finals"]["id"] == runs[0].id  # committed per run


def test_revisits_take_a_bounded_rotating_share_and_age_out(factory):
    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    stuck = [_simple_run(s, "processing", datetime(2026, 10, 25, 0, n)) for n in range(5)]
    old = _simple_run(s, "failed", datetime(2026, 10, 10))
    run_import(factory, sources=["runs"])
    pending = read_cursor(s, "runs")["pending"]
    assert set(pending) == {r.id for r in stuck} | {old.id}
    new = _simple_run(s, "done", datetime(2026, 10, 26))
    report = run_import(factory, sources=["runs"], limit=4)
    assert report["sources"]["runs"]["rows"] == 2  # one revisit, then the new run
    assert store.current_case(s, store.run_case_key(new.id)) is not None
    after = read_cursor(s, "runs")["pending"]
    assert pending[0] == old.id and after == pending[1:]  # the oldest aged out on its revisit
    run_import(factory, sources=["runs"], limit=4)
    assert read_cursor(s, "runs")["pending"] == after[1:] + after[:1]  # rotated, not re-read from the front
    assert any("aged out" in i["what"] for i in report["skipped"]["items"])


def test_full_revisit_list_stops_the_source_instead_of_skipping(factory, monkeypatch):
    from app.knowledge import store
    from app.knowledge.importer import runner
    from app.knowledge.importer.runner import read_cursor

    monkeypatch.setattr(runner, "PENDING_CAP", 1)
    s = factory()
    a = _simple_run(s, "processing", datetime(2026, 10, 25, 1))
    b = _simple_run(s, "processing", datetime(2026, 10, 25, 2))
    first = run_import(factory, sources=["runs"])
    assert read_cursor(s, "runs")["id"] == a.id and first["sources"]["runs"]["caught_up"] is False
    assert store.current_case(s, store.run_case_key(b.id)) is None  # rolled back, not half-imported
    a.status = "done"
    s.commit()
    run_import(factory, sources=["runs"])
    assert read_cursor(s, "runs")["pending"] == [b.id]
    assert store.current_case(s, store.run_case_key(b.id)) is not None


def test_revised_settlement_appends_a_newer_settled_cost(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.models import KieSpendDraw

    s = factory()
    _, _, (t1, *_) = build_world(s)
    run_import(factory)
    draw = s.execute(select(KieSpendDraw).where(KieSpendDraw.call_key == t1)).scalar_one()
    attempt = attempt_by_task(s, t1)
    for n, amount in enumerate((13000, 12000), start=1):
        draw.mcredits, draw.settled_at = amount, datetime(2026, 10, 21, n)
        s.commit()
        report = run_import(factory, sources=["reconcile"])
        assert report["reconcile"]["settlements_recorded"] == 1
        state = store.attempt_state(s, attempt.id)
        assert state["settled_mcredits"] == amount
        assert len([o for o in state["observations"] if o.kind == "cost_settled"]) == n + 1
    assert run_import(factory, sources=["reconcile"])["reconcile"]["settlements_recorded"] == 0


# ---------------------------------------------------------------- Terra review regressions (AICP-297 r3)


def test_final_reconciliation_rechecks_the_run_under_its_lock(factory):
    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor
    from app.models import Run

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/m/final.mp4")
    moved = []

    def hasher(path):  # a promotion re-points the Run while the old final is being hashed
        if not moved:
            with factory() as other:
                other.get(Run, run.id).result_local_path = "/data/m/promoted.mp4"
                other.commit()
            moved.append(path)
        return sha(path)

    first = run_import(factory, sources=["reconcile"], hasher=hasher, exists=lambda p: True)
    assert first["reconcile"]["finals_runs_changed_retry"] == 1 and "finals_queued" not in first["reconcile"]
    assert not (read_cursor(s, "reconcile") or {}).get("finals")
    second = run_import(factory, sources=["reconcile"], hasher=hasher, exists=lambda p: True)
    assert second["reconcile"]["finals_queued"] == 1
    case = store.current_case(s, store.run_case_key(run.id))
    assert [v.source_record for v in store.versions(s, case.id)] == [
        f"reconcile:{run.id}:{sha('/data/m/promoted.mp4')}:0"]


def test_final_reconciliation_skips_a_locked_run_without_advancing(factory):
    from sqlalchemy import select

    from app.knowledge.importer.runner import read_cursor
    from app.models import Run

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/l/final.mp4")
    holder = factory()
    holder.execute(select(Run.id).where(Run.id == run.id).with_for_update())
    blocked = run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: True)
    holder.rollback()
    holder.close()
    assert blocked["reconcile"]["finals_runs_locked_retry"] == 1
    assert not (read_cursor(s, "reconcile") or {}).get("finals")
    assert run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: True)["reconcile"][
        "finals_queued"] == 1


def test_final_reconciliation_retries_failures_then_records_the_miss(factory):
    from sqlalchemy import select

    from app.knowledge.importer.runner import read_cursor
    from app.knowledge.models import KbArtifact, KbArtifactCheck

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/e/final.mp4")

    def broken(path):
        raise OSError("disk hiccup")

    for attempt in (1, 2):
        run_import(factory, sources=["reconcile"], hasher=broken, exists=lambda p: True)
        assert read_cursor(s, "reconcile")["finals"]["failures"] == {run.id: attempt}
        assert "id" not in read_cursor(s, "reconcile")["finals"]
    third = run_import(factory, sources=["reconcile"], hasher=broken, exists=lambda p: True)
    assert any("not reconciled" in i["what"] for i in third["missing_media"]["items"])
    art = s.execute(select(KbArtifact).where(KbArtifact.locator == run.result_local_path)).scalar_one()
    assert art.sha256 is None
    check = s.execute(select(KbArtifactCheck).where(KbArtifactCheck.artifact_id == art.id)).scalar_one()
    assert (check.present, check.sha256_matches) == (True, None)
    # A transient failure is retried on the next pass rather than skipped to the wrap.
    other = _simple_run(s, "done", datetime(2026, 10, 19, 1), result_local_path="/data/e2/final.mp4")
    flaky = {"n": 0}

    def once(path):
        flaky["n"] += 1
        if flaky["n"] == 1:
            raise OSError("transient")
        return sha(path)

    run_import(factory, sources=["reconcile"], hasher=once, exists=lambda p: True)
    assert run_import(factory, sources=["reconcile"], hasher=once, exists=lambda p: True)["reconcile"][
        "finals_queued"] >= 1
    assert other.id


def test_drain_is_bounded_per_call(factory):
    from app.knowledge import store

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19))
    store.ensure_case(s, case_key=store.run_case_key(run.id), scope="run", operation="face_swap",
                      project_id=run.project_id, run_id=run.id)
    for n in range(3):
        art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path", locator=f"/f{n}",
                                           sha256=sha(f"f{n}"), project_id=run.project_id)
        store.enqueue_final_capture(s, run_id=run.id, artifact_id=art.id, reason="restitch",
                                    publication_id=f"stitch:{run.id}:{n}", file_identity="test-live")
    s.commit()
    outcomes = []
    for _ in range(4):
        outcomes.append([d.outcome for d in store.drain_run_captures(s, run.id, max_items=1)])
        s.commit()
    assert outcomes == [["appended"], ["appended"], ["appended"], []]


def test_source_rows_skip_a_locked_run_without_advancing(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor
    from app.models import Run

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19))
    holder = factory()
    holder.execute(select(Run.id).where(Run.id == run.id).with_for_update())
    blocked = run_import(factory, sources=["runs"])
    holder.rollback()
    holder.close()
    assert blocked["sources"]["runs"]["locked"] == 1 and blocked["sources"]["runs"]["caught_up"] is False
    assert (read_cursor(s, "runs") or {}).get("id") is None
    assert store.current_case(s, store.run_case_key(run.id)) is None
    run_import(factory, sources=["runs"])
    assert store.current_case(s, store.run_case_key(run.id)) is not None


def test_only_runs_age_out_of_the_revisit_list(factory):
    from app.knowledge.importer.runner import read_cursor
    from app.models import RunSegment

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 9))
    seg = RunSegment(run_id=run.id, segment_def_id="sd", index=0, status="generating", seedance_task_id="tk-old",
                     created_at=datetime(2026, 10, 9, 1))
    s.add(seg)
    s.commit()
    run_import(factory, sources=["runs", "run_segments"])
    run_import(factory, sources=["runs", "run_segments"])  # a full revisit pass, 18 days later
    assert read_cursor(s, "run_segments")["pending"] == [seg.id]
    seg.status, seg.local_result_path = "completed", "/data/old/result.mp4"
    s.commit()
    run_import(factory, sources=["runs", "run_segments"])
    assert read_cursor(s, "run_segments")["pending"] == []
    assert store_attempt_status(s, "tk-old") == "succeeded"


def store_attempt_status(s, task_id):
    from app.knowledge import store

    return store.attempt_state(s, attempt_by_task(s, task_id).id)["status"]


def test_small_budgets_alternate_revisits_and_new_rows(factory):
    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    stuck = _simple_run(s, "processing", datetime(2026, 10, 25))
    run_import(factory, sources=["runs"])
    fresh = [_simple_run(s, "done", datetime(2026, 10, 26, n)) for n in range(3)]
    passes = [run_import(factory, sources=["runs"], limit=1)["sources"]["runs"] for _ in range(4)]
    assert [p.get("revisit_done", 0) + p["rows"] for p in passes] == [1, 1, 1, 1]
    assert sum(store.current_case(s, store.run_case_key(r.id)) is not None for r in fresh) == 2
    assert read_cursor(s, "runs")["pending"] == [stuck.id]
    two = run_import(factory, sources=["runs"], limit=2)["sources"]["runs"]
    assert two["rows"] == 2  # one revisit and the last new run
    assert all(store.current_case(s, store.run_case_key(r.id)) is not None for r in fresh)


def test_promotions_map_to_existing_versions_and_superseded_ones_become_historical(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbArtifact, KbFeedbackEvent
    from app.models import RunAudioPatch, RunSegment, RunSegmentRangeAttempt

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/p/patched.mp4")
    base, out = "/data/p/final.mp4", "/data/p/patched.mp4"
    patch = RunAudioPatch(run_id=run.id, donor_run_id=str(uuid.uuid4()), target_start_sec=1.0, target_end_sec=2.0,
                          donor_start_sec=1.0, crossfade_sec=0.02, base_result_path=base, status="done",
                          output_path=out, promoted_at=datetime(2026, 10, 19, 2), created_at=datetime(2026, 10, 19, 1))
    rs = RunSegment(run_id=run.id, segment_def_id="sd", index=0, status="completed", created_at=datetime(2026, 10, 19))
    s.add_all([patch, rs])
    s.flush()
    old = RunSegmentRangeAttempt(run_id=run.id, run_segment_id=rs.id, requested_start_sec=0.0, requested_end_sec=4.0,
                                 base_segment_snapshot=[], candidate_segment_snapshot=[], status="done",
                                 final_local_path="/data/p/range-old.mp4", promoted_at=datetime(2026, 10, 19, 1),
                                 created_at=datetime(2026, 10, 19, 0, 30))
    s.add(old)
    # Sync/live capture already appended both finals before the importer ran.
    case = store.ensure_case(s, case_key=store.run_case_key(run.id), scope="run", operation="face_swap",
                             project_id=run.project_id, run_id=run.id)
    for path, rec in ((base, "sync:a"), (out, "sync:b")):
        art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path", locator=path,
                                           sha256=sha(path), project_id=run.project_id)
        store.append_case_version(s, case, artifact_id=art.id, reason="restitch", source_record=rec)
    s.commit()
    every = dict(hasher=sha, exists=lambda p: True)
    report = run_import(factory, sources=["audio_patches", "range_attempts"], **every)
    assert report["failures"]["total"] == 0, report["failures"]
    modes = {p["publication_id"]: (p["mode"], p["case_version_no"]) for p in report["publications"]}
    assert modes[f"audio_patch:{patch.id}"] == ("matched_existing_version", 2)
    assert modes[f"range_attempt:{old.id}"] == ("historical_version", 3)
    hist_v = next(v for v in store.versions(s, case.id) if v.historical)
    assert hist_v.published_at is not None  # the promotion's own time, not NULL
    assert [v.source_record for v in store.versions(s, case.id)] == ["sync:a", "sync:b", f"range_attempt:{old.id}"]
    assert store.latest_version(s, case.id).source_record == "sync:b"  # history never becomes current
    corr = next(e for e in s.execute(select(KbFeedbackEvent)).scalars() if e.event_type == "correction")
    vs = store.versions(s, case.id)
    assert (corr.before_artifact_id, corr.after_artifact_id) == (vs[0].artifact_id, vs[1].artifact_id)
    assert f"audio_patch:{patch.id} -> case version 2" in corr.note
    hist = s.execute(select(KbArtifact).where(KbArtifact.locator == "/data/p/range-old.mp4")).scalars().all()
    assert len(hist) == 1 and hist[0].sha256 == sha("/data/p/range-old.mp4")
    before = evidence_counts(s)
    run_import(factory, sources=["audio_patches", "range_attempts"], from_start=True, **every)
    assert evidence_counts(s) == before


# ---------------------------------------------------------------- Terra review regressions (AICP-297 r4)


def _run_is_locked(factory, run_id) -> bool:
    from sqlalchemy import select, text
    from sqlalchemy.exc import OperationalError

    from app.models import Run

    with factory() as probe:
        probe.execute(text("SET lock_timeout = '50ms'"))
        try:
            probe.execute(select(Run.id).where(Run.id == run_id).with_for_update(nowait=True))
            return False
        except OperationalError:
            return True
        finally:
            probe.rollback()


def test_no_file_io_while_a_run_is_locked(factory):
    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    pid, rid, _ = build_world(s)
    violations = []

    def guard(fn):
        def wrapped(path):
            if _run_is_locked(factory, rid):
                violations.append(path)
            return fn(path)
        return wrapped

    # stat may run under the lock (the re-verification); hashing and existence probes never do.
    report = run_import(factory, hasher=guard(sha), exists=guard(lambda p: True), stat=lambda p: (1, 1))
    assert report["failures"]["total"] == 0, report["failures"]
    assert store.current_case(s, store.run_case_key(rid)) is not None
    assert violations == []
    assert read_cursor(s, "runs")["id"] == rid


def test_a_file_that_changes_before_the_row_is_written_is_retried(factory):
    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/c/final.mp4")
    calls = {"n": 0}

    def stat(path):  # probe sees (1, 1) twice; the re-check under the lock sees a rewrite
        calls["n"] += 1
        return (1, 1) if calls["n"] <= 2 else (2, 2)

    first = run_import(factory, sources=["runs"], hasher=sha, exists=lambda p: True, stat=stat)
    assert first["sources"]["runs"]["media_changed"] == 1
    assert (read_cursor(s, "runs") or {}).get("id") is None
    assert store.current_case(s, store.run_case_key(run.id)) is None
    second = run_import(factory, sources=["runs"], hasher=sha, exists=lambda p: True, stat=lambda p: (2, 2))
    assert second["sources"]["runs"]["rows"] == 1
    case = store.current_case(s, store.run_case_key(run.id))
    assert len(store.versions(s, case.id)) == 1


def test_the_tick_budget_is_shared_fairly_and_draws_progress(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbAttempt
    from app.models import KieSpendDraw, KieSpendReservation, RunSegment

    s = factory()
    runs = [_simple_run(s, "done", datetime(2026, 10, 19, 1, n)) for n in range(30)]
    for n, r in enumerate(runs):
        s.add(RunSegment(run_id=r.id, segment_def_id="sd", index=0, status="failed", seedance_task_id=f"seg-{n}",
                         created_at=datetime(2026, 10, 19, 1, n, 30)))
    res = KieSpendReservation(scope_key=f"run:{runs[0].id}", day=date(2026, 10, 19), kind="new run",
                              run_id=runs[0].id)
    s.add(res)
    s.flush()
    s.add(KieSpendDraw(call_key="lost-early", reservation_id=res.id, kind="video", estimated_mcredits=5,
                       mcredits=5, created_at=datetime(2026, 10, 18, 18, 0)))  # > COMMIT_LAG before the backlog
    s.commit()
    sources = ["runs", "run_segments", "segment_image_jobs", "range_attempts", "spend_draws"]
    passes = [run_import(factory, sources=sources, limit=6)["sources"] for _ in range(3)]
    for p in passes:
        assert p["runs"]["rows"] >= 1 and p["run_segments"]["rows"] >= 1  # neither starves the other
    assert not passes[-1]["runs"]["caught_up"]  # a backlog remains ...
    ambiguous = [a for a in s.execute(select(KbAttempt)).scalars() if a.target_kind == "spend_draw"]
    assert len(ambiguous) == 1  # ... yet draw coverage (held one commit lag behind the slowest source) moved on


def test_an_unreadable_final_is_recorded_missing_once_per_file_identity(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbArtifact

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/u/final.mp4")
    identity = {"v": (1, 1)}

    def broken(path):
        raise OSError("unreadable")

    every = dict(sources=["reconcile"], hasher=broken, exists=lambda p: True, stat=lambda p: identity["v"])
    for _ in range(6):
        run_import(factory, **every)
    arts = lambda: s.execute(select(KbArtifact).where(KbArtifact.locator == run.result_local_path)).scalars().all()  # noqa: E731
    assert len(arts()) == 1
    identity["v"] = (2, 2)  # the file was rewritten: worth trying again
    for _ in range(3):
        run_import(factory, **every)
    assert len(arts()) == 2


def test_capture_drain_rotates_through_runs(factory, monkeypatch):
    from sqlalchemy import func, select

    from app.knowledge import store
    from app.knowledge.importer import reconcile
    from app.knowledge.models import KbCaptureDone, KbCaptureQueue

    monkeypatch.setattr(reconcile, "DRAIN_MAX_ITEMS", 1)
    s = factory()
    runs = [_simple_run(s, "done", datetime(2026, 10, 19, n)) for n in range(3)]
    for r in runs:
        store.ensure_case(s, case_key=store.run_case_key(r.id), scope="run", operation="face_swap",
                          project_id=r.project_id, run_id=r.id)
    for rnd in range(2):  # interleaved: A1 B1 C1 A2 B2 C2 by c1_seq
        for r in runs:
            art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path",
                                               locator=f"/q/{r.id}/{rnd}", sha256=sha(f"{r.id}{rnd}"),
                                               project_id=r.project_id)
            store.enqueue_final_capture(s, run_id=r.id, artifact_id=art.id, reason="restitch",
                                        publication_id=f"stitch:{r.id}:{rnd}", file_identity="test-live")
    s.commit()

    def drained(run):
        return s.execute(select(func.count()).select_from(KbCaptureDone).join(
            KbCaptureQueue, KbCaptureQueue.id == KbCaptureDone.queue_id).where(KbCaptureQueue.run_id == run.id)
        ).scalar()

    run_import(factory, sources=["reconcile"], limit=2)
    assert [drained(r) for r in runs] == [1, 1, 0]
    run_import(factory, sources=["reconcile"], limit=2)
    assert drained(runs[2]) == 1  # the third run is reached, not A/B again


# ---------------------------------------------------------------- Terra review regressions (AICP-297 r5)


def test_source_exports_are_written_after_the_row_releases_its_run(factory, monkeypatch):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbFeedbackEvent

    s = factory()
    _, rid, _ = build_world(s)
    original, held_at_call = store.record_export_observed, []

    def spy(session, **kw):
        held_at_call.append(_run_is_locked(factory, rid))
        return original(session, **kw)

    monkeypatch.setattr(store, "record_export_observed", spy)
    run_import(factory)
    assert held_at_call and not any(held_at_call)  # key → Run → case, never Run → key
    assert len([e for e in s.execute(select(KbFeedbackEvent)).scalars() if e.event_type == "export_observed"]) == 2


def test_an_export_blocked_by_a_held_run_is_retried_with_its_row(factory, monkeypatch):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor
    from app.knowledge.models import KbFeedbackEvent
    from app.models import Run

    s = factory()
    _, rid, _ = build_world(s)
    original, calls = store.record_export_observed, []

    def blocked_once(session, **kw):
        calls.append(1)
        if len(calls) > 1:
            return original(session, **kw)
        holder = factory()
        holder.execute(select(Run.id).where(Run.id == rid).with_for_update())
        try:
            return original(session, **kw)  # the pipeline holds the Run: times out after 200 ms
        finally:
            holder.rollback()
            holder.close()

    monkeypatch.setattr(store, "record_export_observed", blocked_once)
    run_import(factory, sources=["runs"])
    assert [e[2] for e in read_cursor(s, "runs")["exports"]] == [rid]  # queued durably, not lost
    exports = lambda: [e for e in s.execute(select(KbFeedbackEvent)).scalars()  # noqa: E731
                       if e.event_type == "export_observed"]
    assert exports() == []
    run_import(factory, sources=["runs"])
    assert len(exports()) == 1 and read_cursor(s, "runs")["exports"] == []


def test_export_on_a_derived_case_takes_its_run_lock(factory):
    from sqlalchemy import select, text
    from sqlalchemy.exc import OperationalError

    from app.knowledge import store
    from app.models import Run

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19))
    root = store.ensure_case(s, case_key=store.run_case_key(run.id), scope="run", operation="face_swap",
                             project_id=run.project_id, run_id=run.id)
    tail = store.ensure_case(s, case_key=store.derived_case_key("run_tail_edits", str(uuid.uuid4())),
                             scope="derived", operation="face_swap", project_id=run.project_id, run_id=run.id,
                             parent_case_id=root.id)
    s.commit()
    holder = factory()
    holder.execute(select(Run.id).where(Run.id == run.id).with_for_update())
    s.execute(text("SET lock_timeout = '300ms'"))
    with pytest.raises(OperationalError):
        store.record_export_observed(s, case_id=tail.id, idempotency_key="exp-derived", source_record="t:x",
                                     exported_at=datetime(2026, 10, 21, tzinfo=timezone.utc))
    s.rollback()
    holder.rollback()
    holder.close()


def test_missing_finals_are_recorded_once_per_identity_even_when_it_returns(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbArtifact

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/aba/final.mp4")
    identity = {"v": (1, 1)}

    def broken(path):
        raise OSError("unreadable")

    every = dict(sources=["reconcile"], hasher=broken, exists=lambda p: True, stat=lambda p: identity["v"])
    arts = lambda: s.execute(select(KbArtifact).where(KbArtifact.locator == run.result_local_path)).scalars().all()  # noqa: E731
    for value, expected in (((1, 1), 1), ((2, 2), 2), ((1, 1), 2), ((2, 2), 2)):  # A, B, back to A, B again
        identity["v"] = value
        for _ in range(4):
            run_import(factory, **every)
        assert len(arts()) == expected


# ---------------------------------------------------------------- Terra review regressions (AICP-297 r6)


def _exports(s):
    from sqlalchemy import select

    from app.knowledge.models import KbFeedbackEvent

    return [e for e in s.execute(select(KbFeedbackEvent)).scalars() if e.event_type == "export_observed"]


def test_queued_exports_are_durable_with_the_cursor_advance(factory):
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    _, rid, _ = build_world(s)
    calls = []

    def dies_after_the_row_commit() -> bool:
        calls.append(1)
        return len(calls) == 1  # the row's commit heartbeats; the export flush finds the lease gone

    aborted = run_import(factory, sources=["runs"], keepalive=dies_after_the_row_commit)
    assert aborted["outcome"] == "aborted"
    cur = read_cursor(s, "runs")
    assert cur["id"] == rid and [e[2] for e in cur["exports"]] == [rid] and _exports(s) == []
    run_import(factory, sources=["runs"])
    assert len(_exports(s)) == 1 and read_cursor(s, "runs")["exports"] == []


def test_a_full_export_queue_stops_the_source_instead_of_growing(factory, monkeypatch):
    from app.knowledge import store
    from app.knowledge.importer import runner
    from app.knowledge.importer.runner import read_cursor

    monkeypatch.setattr(runner, "EXPORT_CAP", 0)
    s = factory()
    _, rid, _ = build_world(s)
    report = run_import(factory, sources=["runs"])
    assert any("export list full" in i["what"] for i in report["skipped"]["items"])
    assert (read_cursor(s, "runs") or {}).get("id") is None
    assert store.current_case(s, store.run_case_key(rid)) is None


def test_deferred_exports_respect_the_deadline(factory):
    import time

    from app.knowledge.importer.runner import read_cursor

    s = factory()
    pid, rid, _ = build_world(s)

    def slow(path):
        time.sleep(1.5)
        return sha(path)

    run_import(factory, sources=["runs"], hasher=slow, exists=lambda p: p.endswith("patched.mp4"),
               deadline=datetime.now(timezone.utc) + timedelta(seconds=1))
    assert [e[2] for e in read_cursor(s, "runs")["exports"]] == [rid] and _exports(s) == []
    run_import(factory, sources=["runs"])
    assert len(_exports(s)) == 1 and read_cursor(s, "runs")["exports"] == []



def test_seed_shareable_must_be_a_real_boolean(factory, tmp_path):
    seed_dir = _seed_file(tmp_path, """
        rules:
          - slug: quoted-shareable
            kind: empirical_guidance
            statement: "Generic."
            recommendation: {action: warn, target: {model_key: seedance}}
            confidence: low
            shareable: "false"
            effective_from: "2026-10-26T00:00:00Z"
    """)
    report = run_import(factory, seed_dir=seed_dir)
    assert any(i.get("slug") == "quoted-shareable" and "boolean" in i["error"]
               for i in report["failures"]["items"])


def test_a_deleted_final_is_recorded_missing_once(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbArtifact, KbArtifactCheck

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/gone/final.mp4")
    first = run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: False)
    assert any(i["what"] == "final file missing" for i in first["missing_media"]["items"])
    run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: False, from_start=True)
    arts = s.execute(select(KbArtifact).where(KbArtifact.locator == run.result_local_path)).scalars().all()
    assert len(arts) == 1
    check = s.execute(select(KbArtifactCheck).where(KbArtifactCheck.artifact_id == arts[0].id)).scalar_one()
    assert check.present is False


def test_an_explicit_missing_seed_directory_is_a_failure(factory, tmp_path):
    report = run_import(factory, sources=["seed"], seed_dir=str(tmp_path / "nope"))
    assert any(i.get("source") == "seed" and "missing" in i["error"] for i in report["failures"]["items"])
    empty = tmp_path / "empty"
    empty.mkdir()
    report = run_import(factory, sources=["seed"], seed_dir=str(empty))
    assert any(i.get("source") == "seed" for i in report["failures"]["items"])


def test_missing_records_survive_from_start_and_a_second_disappearance_is_recorded(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbArtifact, KbArtifactCheck

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/flip/final.mp4")

    def broken(path):
        raise OSError("unreadable")

    for _ in range(3):
        run_import(factory, sources=["reconcile"], hasher=broken, exists=lambda p: True,
                   stat=lambda p: (10, 1))
    run_import(factory, sources=["reconcile"], hasher=broken, exists=lambda p: True, stat=lambda p: (10, 1),
               from_start=True)
    arts = lambda: s.execute(select(KbArtifact).where(KbArtifact.locator == run.result_local_path)).scalars().all()
    assert len(arts()) == 1  # --from-start does not repeat the recorded miss
    # Gone, then back (readable), then gone again: the second absence is recorded.
    run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: False, stat=lambda p: None,
               from_start=True)
    run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: True, stat=lambda p: (10, 2),
               from_start=True)
    run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: False, stat=lambda p: None,
               from_start=True)
    s.expire_all()
    checks = [c.present for a in arts() for c in s.execute(
        select(KbArtifactCheck).where(KbArtifactCheck.artifact_id == a.id).order_by(KbArtifactCheck.seq)).scalars()]
    assert checks.count(False) >= 2


def test_a_batch_invalidated_mid_import_aborts_without_advancing(factory, monkeypatch):
    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    first = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/b1/final.mp4")
    second = _simple_run(s, "done", datetime(2026, 10, 19, 1), result_local_path="/data/b2/final.mp4")
    real = store.add_import_member
    calls = {"n": 0}

    def withdraw_after_first_row(session, batch_id, kind, target_id):
        if kind == "case" and target_id and calls["n"] == 1:
            raise store.BatchWithdrawn("batch invalidated meanwhile")
        if kind == "case":
            calls["n"] += 1
        return real(session, batch_id, kind, target_id)

    monkeypatch.setattr(store, "add_import_member", withdraw_after_first_row)
    report = run_import(factory, sources=["runs"], hasher=sha, exists=lambda p: True)
    assert any("BatchWithdrawn" in i["error"] for i in report["failures"]["items"])
    cursor = read_cursor(s, "runs") or {}
    assert cursor.get("id") != second.id  # the second run was not skipped
    monkeypatch.setattr(store, "add_import_member", real)
    run_import(factory, sources=["runs"], hasher=sha, exists=lambda p: True)
    assert store.current_case(s, store.run_case_key(second.id)) is not None
    assert first.id


def test_export_reconciliation_aborts_on_a_withdrawn_batch_without_advancing(factory, monkeypatch):
    from app.knowledge import store
    from app.knowledge.importer.runner import read_cursor

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), result_local_path="/data/xb/final.mp4")
    run_import(factory, sources=["runs"], hasher=sha, exists=lambda p: True)
    before = (read_cursor(s, "reconcile") or {}).get("exports")

    def withdrawn(*a, **k):
        raise store.BatchWithdrawn("batch invalidated meanwhile")

    monkeypatch.setattr(store, "record_export_observed", withdrawn)
    s.execute(__import__("sqlalchemy").text("UPDATE runs SET final_creatives_exported_at = now() - interval '1 day' "
                                            "WHERE id = :i"), {"i": run.id})
    s.commit()
    report = run_import(factory, sources=["reconcile"], hasher=sha, exists=lambda p: True)
    assert any("BatchWithdrawn" in i["error"] for i in report["failures"]["items"])
    s.expire_all()
    assert (read_cursor(s, "reconcile") or {}).get("exports") == before


# ---------------------------------------------------------------- final gate round (19) regressions


def test_a_segment_rerun_in_place_is_imported_again(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbAttemptSubmission
    from app.models import RunSegment

    s = factory()
    _, rid, _ = build_world(s)
    run_import(factory, hasher=sha, exists=lambda p: True)
    seg = s.execute(select(RunSegment).where(RunSegment.run_id == rid, RunSegment.index == 0)).scalar_one()
    seg.seedance_task_id = f"task-{rid[:8]}-rerun"  # /rerun resets the row in place; updated_at moves
    seg.status = "completed"
    s.commit()
    run_import(factory, hasher=sha, exists=lambda p: True)
    tasks = {sub.provider_task_id for sub in s.execute(select(KbAttemptSubmission)).scalars()}
    assert f"task-{rid[:8]}-rerun" in tasks and f"task-{rid[:8]}-1" in tasks


def test_a_late_preprocess_on_an_aged_out_run_is_imported(factory):
    from sqlalchemy import select

    from app.knowledge.models import KbAttempt
    from app.models import Run

    s = factory()
    run = _simple_run(s, "preprocess_review", datetime(2026, 10, 1), preprocess_task_ids=["pre-1"])
    run_import(factory, sources=["runs"])
    r = s.get(Run, run.id)
    r.preprocess_task_ids = ["pre-2"]  # re-preprocessing resets the ids in place
    r.status = "done"
    s.commit()
    run_import(factory, sources=["runs"])
    targets = {a.recipe_context.get("reference_index") for a in s.execute(
        select(KbAttempt).where(KbAttempt.target_id == run.id, KbAttempt.attempt_kind == "preprocess_image")).scalars()}
    assert targets  # the new preprocess task produced an attempt
    keys = [a.idempotency_key for a in s.execute(select(KbAttempt).where(KbAttempt.target_id == run.id)).scalars()]
    assert len(keys) >= 2


def test_a_draw_placeholder_is_superseded_by_the_row_that_names_its_task(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt
    from app.models import KieSpendDraw, KieSpendReservation, RunSegment

    s = factory()
    run = _simple_run(s, "done", datetime(2026, 10, 19), segment_snapshot=[
        {"id": "sd-z", "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 4.0}])
    res = KieSpendReservation(scope_key=f"run:{run.id}", day=date(2026, 10, 19), kind="new run", run_id=run.id)
    s.add(res)
    s.flush()
    s.add(KieSpendDraw(call_key="tk-late", reservation_id=res.id, kind="video", estimated_mcredits=100,
                       mcredits=100, created_at=datetime(2026, 10, 19, 0, 1)))
    s.commit()
    run_import(factory)  # no row names tk-late yet: a draw-only placeholder
    holder = s.execute(select(KbAttempt).where(KbAttempt.target_kind == "spend_draw")).scalar_one()
    s.add(RunSegment(run_id=run.id, segment_def_id="sd-z", index=0, status="completed", seedance_task_id="tk-late",
                     created_at=datetime(2026, 10, 19, 0, 1), updated_at=datetime(2026, 10, 19, 0, 2)))
    s.commit()
    run_import(factory, from_start=True)
    s.expire_all()
    invalid = store.InvalidSet.load(s)
    assert invalid.has("attempt", holder.id)
    real = [a for a in s.execute(select(KbAttempt).where(KbAttempt.target_kind == "run_segment")).scalars()]
    assert real and not invalid.attempt_invalid(real[0])


def test_a_segment_rerun_gets_its_own_output_version(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbArtifact, KbAttemptObservation, KbAttemptSubmission
    from app.models import RunSegment

    s = factory()
    _, rid, _ = build_world(s)
    content = {}
    hasher = lambda p: content.get(p, sha(p))  # noqa: E731
    run_import(factory, hasher=hasher, exists=lambda p: True)
    seg = s.execute(select(RunSegment).where(RunSegment.run_id == rid, RunSegment.index == 0)).scalar_one()
    content[seg.local_result_path] = sha("bytes of task B")  # result_0000.mp4 overwritten in place
    seg.seedance_task_id = f"task-{rid[:8]}-B"
    s.commit()
    run_import(factory, hasher=hasher, exists=lambda p: True)
    sub = s.execute(select(KbAttemptSubmission).where(
        KbAttemptSubmission.provider_task_id == f"task-{rid[:8]}-B")).scalar_one()
    out = s.execute(select(KbAttemptObservation).where(KbAttemptObservation.attempt_id == sub.attempt_id,
                                                       KbAttemptObservation.kind == "output")).scalar_one()
    assert s.get(KbArtifact, out.output_artifact_id).sha256 == sha("bytes of task B")
