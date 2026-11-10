"""Knowledge importer adapters on PostgreSQL: assistant promotions, overlays, manual creatives,
segment image jobs and range attempts (AICP-297; design docs/knowledge-feedback-design.md §6.3).

Same opt-in and per-test disposable database as test_knowledge_importer_pg.py. Source rows are
synthetic ORM rows. Assistant promotion receipts sit behind a five-table FK chain the importer
never reads, so they are inserted with ``session_replication_role = replica`` (FK triggers off,
CHECK constraints still on) inside the inserting transaction only.

Every adapter test also asserts the cross-cutting contract: no ``operator_verdict`` event ever,
every created kb_ row registered in ``kb_import_members`` under a valid kind, rerun creates
nothing, and a dry run leaves zero kb_ rows.
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from datetime import date, datetime, timedelta

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)")

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_kb_impad_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_knowledge_importer_pg import (  # noqa: E402,F401  (fixture + helpers reused verbatim)
    _stub_file_identity, evidence_counts, factory, kb_counts, run_import, sha,
)

T0 = datetime(2026, 10, 20, 12, 0, 0)


def _project(s):
    from app.models import VideoProject

    pid = str(uuid.uuid4())
    s.add(VideoProject(id=pid, source_type="upload", source_ref="fixture", project_type="face_swap"))
    s.flush()
    return pid


def make_run(s, pid=None, *, final="final.mp4", gdrive=True, snapshot=True):
    from app.models import Run

    pid = pid or _project(s)
    run = Run(project_id=pid, model="seedance-2-5", status="done", language="es", created_at=T0,
              result_local_path=f"/data/{pid}/{final}" if final else None,
              result_gdrive_file_id=("drive-" + pid[:8]) if gdrive else None,
              segment_snapshot=[{"id": "sd-a", "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 5.0}]
              if snapshot else None)
    s.add(run)
    s.flush()
    return pid, run


def make_segment(s, run, *, status="completed"):
    from app.models import RunSegment

    rs = RunSegment(run_id=run.id, segment_def_id="sd-a", index=0, status=status,
                    created_at=T0 + timedelta(minutes=1))
    s.add(rs)
    s.flush()
    return rs


def make_promotion(s, run_id, *, before, after, noop=False, at=None):
    from sqlalchemy import text

    from app.assistant_models import AssistantRunResultPromotion

    s.execute(text("SET LOCAL session_replication_role = replica"))
    p = AssistantRunResultPromotion(
        id=str(uuid.uuid4()), conversation_id=str(uuid.uuid4()), run_id=run_id, version_id=str(uuid.uuid4()),
        artifact_id=str(uuid.uuid4()), idempotency_key=uuid.uuid4().hex, request_sha256=sha("req"), actor="op",
        before_json=before, after_json=after, delivery="local_only", noop=noop,
        created_at=at or T0 + timedelta(hours=2))
    s.add(p)
    s.flush()
    return p


# ---------------------------------------------------------------- contract helpers


def all_events(s):
    from sqlalchemy import select

    from app.knowledge.models import KbFeedbackEvent

    return s.execute(select(KbFeedbackEvent)).scalars().all()


def assert_contract(s):
    """No operator verdicts; every kb row created by an import is registered under a valid kind."""
    from sqlalchemy import select

    from app.knowledge import taxonomy as t
    from app.knowledge.models import (
        KbArtifact, KbAttempt, KbAttemptObservation, KbAttemptSubmission, KbCase, KbCaseVersion, KbEvalMembership,
        KbFeedbackEvent, KbImportMember,
    )

    assert not [e for e in all_events(s) if e.event_type == "operator_verdict"]
    members = {(m.target_kind, m.target_id) for m in s.execute(select(KbImportMember)).scalars()}
    assert all(kind in t.IMPORT_MEMBER_KINDS for kind, _ in members)
    for model, kind in ((KbCase, "case"), (KbCaseVersion, "case_version"), (KbArtifact, "artifact"),
                        (KbAttempt, "attempt"), (KbAttemptSubmission, "submission"),
                        (KbAttemptObservation, "observation"), (KbFeedbackEvent, "event"),
                        (KbEvalMembership, "eval_membership")):
        missing = [r.id for r in s.execute(select(model)).scalars() if (kind, r.id) not in members]
        assert not missing, f"{kind} rows not registered in kb_import_members: {missing}"


def assert_dry_run_clean(factory):
    assert all(n == 0 for n in kb_counts(factory()).values())
    report = run_import(factory, apply=False)
    assert report["dry_run"] is True and report["outcome"] == "finished"
    assert all(n == 0 for n in kb_counts(factory()).values())
    return report


def assert_rerun_idempotent(factory):
    s = factory()
    before = evidence_counts(s)
    again = run_import(factory)
    assert again["failures"]["total"] == 0 and again["created"] == {}, again["created"]
    full = run_import(factory, from_start=True)
    assert full["failures"]["total"] == 0 and full["created"] == {}, (full["failures"], full["created"])
    assert evidence_counts(factory()) == before


def case_by_key(s, key):
    from app.knowledge import store

    return store.current_case(s, key)


def version_rows(s, case):
    from app.knowledge import store
    from app.knowledge.models import KbArtifact

    return [(v.reason, v.source_record, s.get(KbArtifact, v.artifact_id)) for v in store.versions(s, case.id)]


# ---------------------------------------------------------------- 1. assistant promotions


def test_assistant_promotion_correction_uses_recorded_shas(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttemptObservation

    s = factory()
    pid, run = make_run(s, final="after.mp4")
    b_sha, a_sha = sha("before-bytes"), sha("after-bytes")
    p = make_promotion(s, run.id, before={"path": f"/data/{pid}/before.mp4", "sha256": b_sha},
                       after={"path": f"/data/{pid}/after.mp4", "sha256": a_sha})
    s.commit()

    assert_dry_run_clean(factory)
    report = run_import(factory)
    assert report["outcome"] == "finished" and report["failures"]["total"] == 0, report["failures"]
    s = factory()
    case = case_by_key(s, store.run_case_key(run.id))
    rows = version_rows(s, case)
    assert [(r, src) for r, src, _ in rows] == [("import", f"import:before:assistant_promotion:{p.id}"),
                                                  ("assistant_promotion", f"assistant_promotion:{p.id}")]
    assert [(a.locator, a.sha256) for _, _, a in rows] == [(f"/data/{pid}/before.mp4", b_sha),
                                                            (f"/data/{pid}/after.mp4", a_sha)]
    corrections = [e for e in all_events(s) if e.event_type == "correction"]
    assert len(corrections) == 1
    c = corrections[0]
    assert (c.before_artifact_id, c.after_artifact_id) == (rows[0][2].id, rows[1][2].id)
    assert c.method == "assistant" and c.resolved == "unknown" and c.actor_kind == "importer"
    assert c.case_id == case.id
    attempt_obs = s.execute(select(KbAttemptObservation).where(KbAttemptObservation.kind == "output")).scalars().all()
    assert any(o.output_artifact_id == rows[1][2].id for o in attempt_obs)
    assert_contract(s)
    assert_rerun_idempotent(factory)
    assert len([e for e in all_events(factory()) if e.event_type == "correction"]) == 1


def test_assistant_promotion_alone_without_recorded_before_version_reports_no_correction(factory):
    from app.knowledge import store

    s = factory()
    pid, run = make_run(s, final="after.mp4")
    p = make_promotion(s, run.id, before={"path": f"/data/{pid}/before.mp4", "sha256": sha("b")},
                       after={"path": f"/data/{pid}/after.mp4", "sha256": sha("a")})
    s.commit()
    # Only the promotion adapter: the Run adapter never built the before version.
    report = run_import(factory, sources=["assistant_promotions"])
    assert report["failures"]["total"] == 0, report["failures"]
    s = factory()
    assert not [e for e in all_events(s) if e.event_type == "correction"]
    assert any(i.get("source") == f"assistant_run_result_promotions:{p.id}"
               for i in report["missing_metadata"]["items"])
    rows = version_rows(s, case_by_key(s, store.run_case_key(run.id)))
    assert [r for r, _, _ in rows] == ["assistant_promotion"]
    assert_contract(s)
    assert_rerun_idempotent(factory)


def test_assistant_noop_promotion_is_a_status_observation_only(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt

    s = factory()
    pid, run = make_run(s)
    path = f"/data/{pid}/final.mp4"
    p = make_promotion(s, run.id, before={"path": path, "sha256": sha("x")}, after={"path": path, "sha256": sha("x")},
                       noop=True)
    s.commit()
    assert_dry_run_clean(factory)
    report = run_import(factory)
    assert report["failures"]["total"] == 0, report["failures"]
    s = factory()
    attempt = next(a for a in s.execute(select(KbAttempt)).scalars() if a.target_id == p.id)
    assert attempt.attempt_kind == "run_result_promotion" and attempt.origin == "assistant"
    obs = store.attempt_state(s, attempt.id)["observations"]
    assert [(o.kind, o.status, o.payload) for o in obs if o.kind != "intent"][-1:] == [
        ("status", "succeeded", {"noop": True})]
    assert not [o for o in obs if o.kind == "output"]
    case = case_by_key(s, store.run_case_key(run.id))
    assert all(r != "assistant_promotion" for r, _, _ in version_rows(s, case))  # no version from a no-op
    assert not [e for e in all_events(s) if e.event_type == "correction"]
    assert_contract(s)
    assert_rerun_idempotent(factory)


def test_assistant_promotion_malformed_sources_are_reported_not_fatal(factory):
    s = factory()
    pid, run = make_run(s)
    no_path = make_promotion(s, run.id, before={}, after={"sha256": sha("a")}, at=T0 + timedelta(hours=1))
    ghost = make_promotion(s, str(uuid.uuid4()), before={"path": "/x/b.mp4"}, after={"path": "/x/a.mp4"})
    make_promotion(s, run.id, before={"path": f"/data/{pid}/old.mp4", "sha256": "NOT-HEX"},
                             after={"path": f"/data/{pid}/final.mp4", "sha256": "ZZ"}, at=T0 + timedelta(hours=3))
    s.commit()
    report = run_import(factory)
    assert report["outcome"] == "finished" and report["failures"]["total"] == 0, report["failures"]
    assert any(i.get("promotion_id") == no_path.id for i in report["missing_metadata"]["items"])
    assert any(i.get("promotion_id") == ghost.id for i in report["skipped"]["items"])
    s = factory()
    # A malformed sha is dropped (unknown), never stored; the path chain still resolves by path.
    from sqlalchemy import select

    from app.knowledge.models import KbArtifact

    assert all(a.sha256 is None for a in s.execute(select(KbArtifact)).scalars())
    assert_contract(s)
    assert_rerun_idempotent(factory)


# ---------------------------------------------------------------- 2. overlays


def make_overlay(s, **kw):
    from app.models import OverlayJob

    kw.setdefault("status", "done")
    kw.setdefault("source_type", "run")
    kw.setdefault("created_at", T0 + timedelta(hours=5))
    job = OverlayJob(overlays=[{"asset": "a"}], **kw)
    s.add(job)
    s.flush()
    return job


def test_overlay_is_a_derived_case_under_the_run_case(factory):
    from app.knowledge import store

    s = factory()
    pid, run = make_run(s)
    job = make_overlay(s, parent_run_id=run.id, result_local_path=f"/data/{pid}/overlay.mp4",
                       gdrive_file_id="drive-ov")
    deleted = make_overlay(s, source_type="run", source_run_id=run.id, status="failed", error="ffmpeg boom",
                           deleted_at=T0 + timedelta(hours=9), created_at=T0 + timedelta(hours=6))
    s.commit()
    assert_dry_run_clean(factory)
    report = run_import(factory)
    assert report["failures"]["total"] == 0, report["failures"]
    s = factory()
    parent = case_by_key(s, store.run_case_key(run.id))
    dcase = case_by_key(s, store.derived_case_key("overlay_jobs", job.id))
    assert dcase.scope == "derived" and dcase.parent_case_id == parent.id and dcase.run_id == run.id
    assert dcase.language == "es" and dcase.origin == "import" and "reason_for_edit" in dcase.missing_facts
    rows = version_rows(s, dcase)
    assert [(r, a.kind, a.locator) for r, _, a in rows] == [("import", "overlay", f"/data/{pid}/overlay.mp4")]
    fcase = case_by_key(s, store.derived_case_key("overlay_jobs", deleted.id))
    assert "output_media_deleted" in fcase.missing_facts and not version_rows(s, fcase)
    assert_contract(s)
    assert_rerun_idempotent(factory)


def test_overlay_source_type_chain_and_unresolvable_upload_is_skipped(factory):
    from app.knowledge import store
    from app.models import RunTailEdit

    s = factory()
    pid, run = make_run(s)
    te = RunTailEdit(run_id=run.id, cut_sec=3.0, source_type="run", status="done", output_path=f"/data/{pid}/t.mp4",
                     created_at=T0 + timedelta(hours=3))
    s.add(te)
    s.flush()
    via_tail = make_overlay(s, source_type="tail_edit", source_tail_edit_id=te.id,
                            result_local_path=f"/data/{pid}/o1.mp4", gdrive_file_id="d1")
    upload = make_overlay(s, source_type="upload", source_local_path="/data/up.mp4", gdrive_file_id="d2",
                          created_at=T0 + timedelta(hours=7))
    dangling = make_overlay(s, source_type="caption", source_caption_id=str(uuid.uuid4()), gdrive_file_id="d3",
                            created_at=T0 + timedelta(hours=8))
    s.commit()
    report = run_import(factory)
    assert report["outcome"] == "finished" and report["failures"]["total"] == 0, report["failures"]
    skipped = {i["overlay_id"] for i in report["skipped"]["items"] if "overlay_id" in i}
    assert skipped == {upload.id, dangling.id}
    s = factory()
    assert case_by_key(s, store.derived_case_key("overlay_jobs", via_tail.id)).run_id == run.id
    assert case_by_key(s, store.derived_case_key("overlay_jobs", upload.id)) is None
    assert_contract(s)
    assert_rerun_idempotent(factory)


# ---------------------------------------------------------------- 3. manual creatives


def make_manual(s, **kw):
    from app.models import ManualCreative

    kw.setdefault("gdrive_file_id", "drive-" + uuid.uuid4().hex[:10])
    kw.setdefault("created_at", T0 + timedelta(hours=10))
    mc = ManualCreative(pipeline_type="re-skin-manual-script", idea="TenBooks", language="fr", **kw)
    s.add(mc)
    s.flush()
    return mc


def test_manual_creative_resolves_through_overlay_to_the_run(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt

    s = factory()
    pid, run = make_run(s)
    job = make_overlay(s, parent_run_id=run.id, result_local_path=f"/data/{pid}/o.mp4", gdrive_file_id="d-ov")
    mc = make_manual(s, overlay_job_id=job.id, linear_issue="AICP-238", gdrive_file_id="drive-manual-1")
    s.commit()
    assert_dry_run_clean(factory)
    report = run_import(factory)
    assert report["failures"]["total"] == 0, report["failures"]
    s = factory()
    parent = case_by_key(s, store.run_case_key(run.id))
    case = case_by_key(s, store.manual_case_key(mc.id))
    assert case.scope == "manual" and case.parent_case_id == parent.id and case.language == "fr"
    assert case.project_id == pid and {"script_recipe", "reason_for_edit"} <= set(case.missing_facts)
    rows = version_rows(s, case)
    assert [(r, a.kind, a.locator_kind, a.locator) for r, _, a in rows] == [
        ("import", "delivered_final", "drive_file", "drive-manual-1")]
    attempt = next(a for a in s.execute(select(KbAttempt)).scalars() if a.target_id == mc.id)
    assert attempt.attempt_kind == "manual_script" and attempt.origin == "operator"
    assert any(o.kind == "output" and o.output_artifact_id == rows[0][2].id
               for o in store.attempt_state(s, attempt.id)["observations"])
    assert_contract(s)
    assert_rerun_idempotent(factory)


def test_manual_creative_without_project_is_skipped_and_reported(factory):
    s = factory()
    pid, run = make_run(s)
    upload_job = make_overlay(s, source_type="upload", source_local_path="/data/u.mp4", gdrive_file_id="d-up")
    no_overlay = make_manual(s, linear_issue="AICP-1")
    dead_overlay = make_manual(s, overlay_job_id=str(uuid.uuid4()), linear_issue="AICP-2",
                               created_at=T0 + timedelta(hours=11))
    via_upload = make_manual(s, overlay_job_id=upload_job.id, linear_issue="AICP-3",
                             created_at=T0 + timedelta(hours=12))
    s.commit()
    report = run_import(factory)
    assert report["outcome"] == "finished" and report["failures"]["total"] == 0, report["failures"]
    skipped = {i["manual_creative_id"] for i in report["skipped"]["items"] if "manual_creative_id" in i}
    assert skipped == {no_overlay.id, dead_overlay.id, via_upload.id}
    s = factory()
    from sqlalchemy import select

    from app.knowledge.models import KbCase

    assert not s.execute(select(KbCase).where(KbCase.scope == "manual")).scalars().all()
    assert_contract(s)
    assert_rerun_idempotent(factory)


# ---------------------------------------------------------------- 4. segment image jobs


def make_image_job(s, run, rs, **kw):
    from app.models import RunSegmentImageJob

    kw.setdefault("status", "done")
    kw.setdefault("created_at", T0 + timedelta(hours=1))
    job = RunSegmentImageJob(run_id=run.id, run_segment_id=rs.id, frame_sec=2.5, prompt="p", **kw)
    s.add(job)
    s.flush()
    return job


def test_segment_image_job_attempt_with_draw_cost_and_labeled_snapshot_cost(factory):
    from app.knowledge import store
    from app.models import KieSpendDraw, KieSpendReservation
    from sqlalchemy import select

    from app.knowledge.models import KbAttempt

    s = factory()
    pid, run = make_run(s)
    rs = make_segment(s, run)
    res = KieSpendReservation(scope_key=f"segimg:x", day=date(2026, 10, 20), kind="segment image", run_id=run.id)
    s.add(res)
    s.flush()
    s.add(KieSpendDraw(call_key="img-task-1", reservation_id=res.id, kind="image", estimated_mcredits=3000,
                       mcredits=3500, created_at=T0, settled_at=T0 + timedelta(minutes=5)))
    drawn = make_image_job(s, run, rs, kie_task_id="img-task-1", output_path=f"/data/{run.id}/img1.png",
                           input_paths=["/a.png", "/b.png"], cost_credits=99.0)
    snap = make_image_job(s, run, rs, kie_task_id="img-task-2", output_path=f"/data/{run.id}/img2.png",
                          cost_credits=2.5, created_at=T0 + timedelta(hours=2))
    failed = make_image_job(s, run, rs, status="failed", error_message="moderated",
                            created_at=T0 + timedelta(hours=3))
    s.commit()
    assert_dry_run_clean(factory)
    report = run_import(factory)
    assert report["failures"]["total"] == 0, report["failures"]
    s = factory()
    seg = case_by_key(s, store.segment_case_key(run.id, "sd-a"))
    assert seg.scope == "segment" and seg.parent_case_id == case_by_key(s, store.run_case_key(run.id)).id
    attempts = {a.target_id: a for a in s.execute(select(KbAttempt)).scalars() if a.target_kind == "segment_image_job"}
    assert set(attempts) == {drawn.id, snap.id, failed.id}
    assert all(a.case_id == seg.id and a.attempt_kind == "segment_image" and a.origin == "operator"
               for a in attempts.values())
    st = store.attempt_state(s, attempts[drawn.id].id)
    assert (st["status"], st["estimated_mcredits"], st["settled_mcredits"], st["cost_source"]) == (
        "succeeded", 3000, 3500, "spend_draw")
    st2 = store.attempt_state(s, attempts[snap.id].id)
    cost = [o for o in st2["observations"] if o.kind == "cost_settled"]
    assert [(o.mcredits, o.cost_source, o.payload["label"], o.payload["column"]) for o in cost] == [
        (2500, "run_segment_snapshot", "current attempt at import time", "run_segment_image_jobs.cost_credits")]
    out = [o for o in st2["observations"] if o.kind == "output"]
    assert len(out) == 1
    assert store.attempt_state(s, attempts[failed.id].id)["status"] == "failed"
    assert_contract(s)
    assert_rerun_idempotent(factory)


def test_segment_image_job_missing_output_and_cost_is_tolerated(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt

    s = factory()
    pid, run = make_run(s, snapshot=False)  # no segment_snapshot: segment time range unknown
    rs = make_segment(s, run)
    job = make_image_job(s, run, rs, status="done", output_path=None, kie_task_id=None, input_paths=None)
    s.commit()
    report = run_import(factory)
    assert report["outcome"] == "finished" and report["failures"]["total"] == 0, report["failures"]
    assert report["missing_metadata"]["total"] >= 1  # segment not in the snapshot
    s = factory()
    attempt = next(a for a in s.execute(select(KbAttempt)).scalars() if a.target_id == job.id)
    st = store.attempt_state(s, attempt.id)
    assert not [o for o in st["observations"] if o.kind in ("output", "cost_settled")]
    assert "time_range" in case_by_key(s, store.segment_case_key(run.id, "sd-a")).missing_facts
    assert_contract(s)
    assert_rerun_idempotent(factory)


# ---------------------------------------------------------------- 5. range attempts


def make_range(s, run, rs, **kw):
    from app.models import RunSegmentRangeAttempt

    kw.setdefault("status", "done")
    kw.setdefault("created_at", T0 + timedelta(hours=1))
    ra = RunSegmentRangeAttempt(run_id=run.id, run_segment_id=rs.id, requested_start_sec=0.0, requested_end_sec=6.0,
                                base_segment_snapshot=[], candidate_segment_snapshot=[], **kw)
    s.add(ra)
    s.flush()
    return ra


def test_promoted_range_attempt_adds_segment_and_run_versions_without_a_correction(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt

    s = factory()
    pid, run = make_run(s, final="range_final.mp4")
    rs = make_segment(s, run)
    ra = make_range(s, run, rs, seedance_task_id="range-task-1", cost_credits=4.0,
                    local_result_path=f"/data/{pid}/range_out.mp4", final_local_path=f"/data/{pid}/range_final.mp4",
                    promoted_at=T0 + timedelta(hours=2))
    s.commit()
    assert_dry_run_clean(factory)
    report = run_import(factory)
    assert report["failures"]["total"] == 0, report["failures"]
    s = factory()
    seg = case_by_key(s, store.segment_case_key(run.id, "sd-a"))
    runc = case_by_key(s, store.run_case_key(run.id))
    pub = f"range_attempt:{ra.id}"
    seg_rows = version_rows(s, seg)
    assert [(r, src, a.kind, a.locator) for r, src, a in seg_rows] == [
        ("range_attempt", pub, "model_output", f"/data/{pid}/range_out.mp4")]
    run_rows = version_rows(s, runc)
    assert [(r, src, a.locator) for r, src, a in run_rows] == [("range_attempt", pub, f"/data/{pid}/range_final.mp4")]
    attempt = next(a for a in s.execute(select(KbAttempt)).scalars() if a.target_id == ra.id)
    assert attempt.case_id == seg.id and attempt.attempt_kind == "range_attempt" and attempt.origin == "operator"
    assert attempt.model_key == "seedance-2-5"
    st = store.attempt_state(s, attempt.id)
    assert st["status"] == "succeeded" and st["provider_task_id"] == "range-task-1"
    assert [(o.mcredits, o.cost_source) for o in st["observations"] if o.kind == "cost_settled"] == [
        (4000, "run_segment_snapshot")]
    assert not [e for e in all_events(s) if e.event_type == "correction"]
    assert any(i.get("range_attempt_id") == ra.id and "no correction" in i["what"]
               for i in report["missing_metadata"]["items"])
    assert_contract(s)
    assert_rerun_idempotent(factory)


def test_range_attempt_failed_and_superseded_promotion(factory):
    from sqlalchemy import select

    from app.knowledge import store
    from app.knowledge.models import KbAttempt

    s = factory()
    pid, run = make_run(s, final="current.mp4")
    rs = make_segment(s, run)
    failed = make_range(s, run, rs, status="failed", error_message="kie down", seedance_task_id="range-task-f")
    # Promoted, but the Run's final has since moved on.
    stale = make_range(s, run, rs, local_result_path=f"/data/{pid}/o.mp4", final_local_path=f"/data/{pid}/older.mp4",
                       promoted_at=T0 + timedelta(hours=2), created_at=T0 + timedelta(hours=2))
    s.commit()
    # Range adapter alone: the older promotion becomes a historical version, never current.
    alone = run_import(factory, sources=["range_attempts"])
    assert alone["failures"]["total"] == 0, alone["failures"]
    assert any(i.get("publication_id") == f"range_attempt:{stale.id}" for i in alone["historical"]["items"])
    s = factory()
    runc = case_by_key(s, store.run_case_key(run.id))
    assert [(r, src) for r, src, _ in version_rows(s, runc)] == [("range_attempt", f"range_attempt:{stale.id}")]
    assert store.latest_version(s, runc.id) is None  # history only, nothing current yet
    attempt = next(a for a in s.execute(select(KbAttempt)).scalars() if a.target_id == failed.id)
    st = store.attempt_state(s, attempt.id)
    assert st["status"] == "failed" and st["provider_task_id"] == "range-task-f"
    assert_contract(s)

    # Full import: the Run adapter orders the history, so the older promotion is a past version, then the current final.
    report = run_import(factory)
    assert report["failures"]["total"] == 0, report["failures"]
    s = factory()
    assert [(r, src) for r, src, _ in version_rows(s, runc)] == [("range_attempt", f"range_attempt:{stale.id}"),
                                                                  ("import", f"import:runs:{run.id}")]
    assert not [e for e in all_events(s) if e.event_type == "correction"]
    assert_contract(s)
    assert_rerun_idempotent(factory)


def test_unattributable_paid_draw_is_skipped_not_failed(factory):
    """Prod dry run 2026-10-28: 21 such draws failed with a Report.note kwarg clash."""
    from app.models import KieSpendDraw, KieSpendReservation

    s = factory()
    res = KieSpendReservation(scope_key=f"run:{uuid.uuid4()}", day=date(2026, 10, 20), kind="new run")
    s.add(res)
    s.flush()
    s.add(KieSpendDraw(call_key=f"orphan-{uuid.uuid4().hex}", reservation_id=res.id, kind="video",
                       estimated_mcredits=1000, mcredits=1000, created_at=T0, settled_at=T0 + timedelta(minutes=1)))
    s.commit()
    report = run_import(factory, sources=["spend_draws"])
    assert report["failures"]["total"] == 0, report["failures"]
    assert any(i.get("what") == "paid call with no attributable case" and i.get("draw_kind") == "video"
               for i in report["skipped"]["items"])
    assert (report["reconcile"]["unattributable_draws"], report["reconcile"]["unattributable_draws_mcredits"]) == \
        (1, 1000)
    from sqlalchemy import select

    from app.knowledge.models import KbImportBatchObservation

    s = factory()
    stored = s.execute(select(KbImportBatchObservation).where(
        KbImportBatchObservation.batch_id == report["batch_id"])).scalars().all()
    assert [o.report["reconcile"]["unattributable_draws"] for o in stored] == [1]  # durable, not the capped list


def test_unattributable_draws_are_tallied_in_the_cursor_with_each_commit(factory):
    """Durable with the cursor advance: a pass that stops (or dies) after one draw keeps its
    count, the next pass adds the other, and nothing is counted twice."""
    from app.knowledge.importer.runner import read_cursor
    from app.models import KieSpendDraw, KieSpendReservation

    s = factory()
    ids = []
    for n in range(2):
        res = KieSpendReservation(scope_key=f"run:{uuid.uuid4()}", day=date(2026, 10, 20), kind="new run")
        s.add(res)
        s.flush()
        d = KieSpendDraw(call_key=f"gone-{n}-{uuid.uuid4().hex}", reservation_id=res.id, kind="video",
                         estimated_mcredits=500, mcredits=700 + n, created_at=T0 + timedelta(minutes=n),
                         settled_at=T0 + timedelta(minutes=n + 1))
        s.add(d)
        s.flush()
        ids.append(d.id)
    s.commit()

    def tally():
        x = factory()
        try:
            return {k: v for k, v in ((read_cursor(x, "spend_draws") or {}).get("tally") or {}).items()
                    if k.split(":", 1)[1] in ids}
        finally:
            x.close()

    run_import(factory, sources=["spend_draws"], limit=1)  # stops after the first committed draw
    assert tally() == {f"unattributable:{ids[0]}": 700}
    run_import(factory, sources=["spend_draws"])
    run_import(factory, sources=["spend_draws"])  # a rerun is a no-op
    assert tally() == {f"unattributable:{ids[0]}": 700, f"unattributable:{ids[1]}": 701}
