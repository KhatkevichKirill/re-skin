"""AICP-389 on real PostgreSQL row locks: one winner per run result.

The brand-fix admission (``LocalisationApplyBrandAudioPatch.execute``) races a
manual audio patch (``api_v2.create_run_audio_patch``), a range-attempt
admission and a segment-rerun admission (their real gates, under the same
run-row lock). Exactly one wins; the loser is a readable 409 and writes
nothing. A crash between the brand admission's commit and its enqueue leaves
ONE queued row, never two.

Opt in with AUTOPILOT_TEST_POSTGRES_URL pointing at a THROWAWAY database
(never production). Each test gets its own schema with the full metadata.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

_OPT_IN = "AUTOPILOT_TEST_POSTGRES_URL"
FFMPEG = shutil.which("ffmpeg")
pytestmark = [pytest.mark.skipif(not os.environ.get(_OPT_IN),
                                 reason=f"opt in with {_OPT_IN}; throwaway test DB only"),
              pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")]


def _tone(path, seconds=3.0, hz=220):
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y",
                    "-f", "lavfi", "-i", f"color=c=0x405060:s=96x160:r=12:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency={hz}:sample_rate=16000:duration={seconds}",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "64k", str(path)], check=True)
    return str(path)


@pytest.fixture
def pg(monkeypatch, tmp_path):
    import app.api_v2 as api_v2_module
    from app import tasks as tasks_mod
    from app.config import settings
    from app.db import Base

    url = os.environ[_OPT_IN]
    assert "re-skin-db" not in url and "@db" not in url, "never the production database"
    schema = "bf_" + uuid.uuid4().hex[:12]
    admin = create_engine(url, poolclass=NullPool)
    with admin.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, poolclass=NullPool, connect_args={
        "options": f"-c search_path={schema} -c lock_timeout=10000"})
    Base.metadata.create_all(engine)
    SF = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(settings, "AUTOPILOT_BRAND_FIX_ENABLED", True)
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path), raising=False)
    enq = []
    monkeypatch.setattr(api_v2_module, "enqueue_process_audio_patch", enq.append)
    monkeypatch.setattr(tasks_mod, "enqueue_process_audio_patch", enq.append)
    SF.enqueued = enq
    try:
        yield SF
    finally:
        engine.dispose()
        with admin.begin() as c:
            c.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _setup(SF, tmp_path, monkeypatch):
    from app import brand_check as bc
    from app.autopilot.compute import attempt_dir
    from app.models import Run, RunSegment, SegmentDef, VideoProject
    from app.state_machine import RunStatus, SegmentStatus

    db = SF()
    p = VideoProject(source_type="upload", source_ref="t.mp4", status="ready",
                     project_type="localisation", mode="assistant", duration_sec=3.0)
    db.add(p)
    db.flush()
    sd = SegmentDef(project_id=p.id, index=0, start_sec=0.0, end_sec=3.0, action="swap")
    db.add(sd)
    db.flush()
    final = _tone(tmp_path / "final.mp4")
    donor_final = _tone(tmp_path / "donor.mp4", hz=330)
    run = Run(project_id=p.id, prompt="x", status=RunStatus.done, result_local_path=final,
              result_duration_sec=3.0, language="fr",
              segment_snapshot=[{"id": sd.id, "index": 0, "action": "swap", "start_sec": 0.0,
                                 "end_sec": 3.0, "pre_roll_sec": 0.0, "post_roll_sec": 0.0}])
    donor = Run(project_id=p.id, prompt="x", status=RunStatus.done, result_local_path=donor_final,
                result_duration_sec=3.0, language="fr")
    db.add_all([run, donor])
    db.flush()
    rs = RunSegment(run_id=run.id, segment_def_id=sd.id, index=0, status=SegmentStatus.completed,
                    local_result_path=final)
    db.add(rs)
    db.commit()
    eff_id = str(uuid.uuid4())
    out = attempt_dir(p.id, eff_id, "tok")
    os.makedirs(out, exist_ok=True)
    cand = os.path.join(out, "candidate_o1.mp4")
    bc.render_candidate(final, cand, bc.render_params([1.0, 1.2]))
    params = {"run_id": run.id, "occ_id": "o1", "window": [1.0, 1.2],
              "filter_params": bc.render_params([1.0, 1.2]), "base_path": final,
              "base_sha256": bc.sha256_file(final), "candidate_path": cand,
              "candidate_sha256": bc.sha256_file(cand), "effect_id": eff_id}
    ids = (p.id, run.id, donor.id, rs.id)
    db.close()
    return ids, params


def _brand_admission(SF, pid, params, results, gate=None):
    from types import SimpleNamespace

    from app.autopilot.contract import ActionRefused
    from app.autopilot.flows.localisation_brandfix import LocalisationApplyBrandAudioPatch
    from app.models import VideoProject

    db = SF()
    try:
        if gate:
            gate.wait()
        h = LocalisationApplyBrandAudioPatch()
        project = db.get(VideoProject, pid)
        p = h.validate(db, project, params)
        h.execute(db, project, p, SimpleNamespace(created_refs={}, result={}))
        results.append("brand")
    except ActionRefused as exc:
        db.rollback()
        results.append(f"brand-{exc.status}")
    finally:
        db.close()


def _manual_patch(SF, rid, donor_id, results, gate=None):
    from fastapi import HTTPException

    import app.api_v2 as api_v2

    db = SF()
    try:
        if gate:
            gate.wait()
        api_v2.create_run_audio_patch(rid, donor_run_id=donor_id, target_start_sec=0.5,
                                      target_end_sec=1.5, donor_start_sec=0.5, crossfade_sec=None,
                                      db=db)
        results.append("manual")
    except HTTPException as exc:
        db.rollback()
        results.append(f"manual-{exc.status_code}")
    finally:
        db.close()


def _active(SF, rid):
    from app.models import RunAudioPatch

    db = SF()
    try:
        return [p for p in db.execute(select(RunAudioPatch).where(RunAudioPatch.run_id == rid)).scalars()
                if p.status.value in ("queued", "processing", "delivering")]
    finally:
        db.close()


def test_brand_fix_vs_manual_audio_patch_one_winner(pg, tmp_path, monkeypatch):
    (pid, rid, donor, _rs), params = _setup(pg, tmp_path, monkeypatch)
    for _ in range(3):
        results: list = []
        gate = threading.Barrier(2)
        ts = [threading.Thread(target=_brand_admission, args=(pg, pid, params, results, gate)),
              threading.Thread(target=_manual_patch, args=(pg, rid, donor, results, gate))]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        winners = [r for r in results if r in ("brand", "manual")]
        assert len(winners) == 1 and len(results) == 2, results
        assert sorted(r.split("-")[-1] for r in results if r not in winners) == ["409"]
        active = _active(pg, rid)
        assert len(active) == 1
        # settle the winner so the next iteration races again
        db = pg()
        from app.models import RunAudioPatch
        from app.state_machine import AudioPatchStatus

        row = db.get(RunAudioPatch, active[0].id)
        row.status = AudioPatchStatus.failed
        db.commit()
        db.close()


def _hold_lock_then(SF, rid, results, *, action, hold_sec=1.0, started=None):
    """A competing admission: takes the run-row lock through its real gates,
    holds it, then commits its change (the shape of a rerun / range attempt)."""
    from fastapi import HTTPException

    import app.api_v2 as api_v2
    from app.models import Run
    from app.state_machine import RunStatus

    db = SF()
    try:
        run = db.get(Run, rid)
        try:
            if action == "rerun":
                api_v2._admit_segment_rerun(run, db)
                api_v2._check_segment_rerun_conflicts(rid, db)
            else:
                api_v2._lock_run_for_result_mutation(run, db)
                if api_v2._active_range_attempt(rid, db) is not None:
                    raise HTTPException(409, "range attempt active")
                active = api_v2._active_audio_patch(rid, db)
                if active is not None:
                    raise api_v2._audio_patch_conflict(active)
        except HTTPException as exc:
            db.rollback()
            results.append(f"{action}-{exc.status_code}")
            if started:
                started.set()
            return
        if started:
            started.set()
        time.sleep(hold_sec)
        if action == "rerun":
            run.status = RunStatus.queued  # what the rerun's admission commits
        else:
            from app.models import RunSegmentRangeAttempt

            db.add(RunSegmentRangeAttempt(
                run_id=rid, run_segment_id=db.execute(select(__import__(
                    "app.models", fromlist=["RunSegment"]).RunSegment.id).where(
                    __import__("app.models", fromlist=["RunSegment"]).RunSegment.run_id == rid)).scalar_one(),
                requested_start_sec=0.0, requested_end_sec=3.0,
                base_segment_snapshot=run.segment_snapshot, candidate_segment_snapshot=run.segment_snapshot))
        db.commit()
        results.append(action)
    finally:
        db.close()


@pytest.mark.parametrize("action", ["rerun", "range"])
def test_competing_admission_holding_the_lock_wins_and_the_brand_fix_is_refused(
        pg, tmp_path, monkeypatch, action):
    (pid, rid, _d, _rs), params = _setup(pg, tmp_path, monkeypatch)
    results: list = []
    started = threading.Event()
    t = threading.Thread(target=_hold_lock_then, args=(pg, rid, results),
                         kwargs={"action": action, "started": started})
    t.start()
    assert started.wait(10)
    _brand_admission(pg, pid, params, results)  # blocks on the lock, then sees the winner
    t.join(30)
    assert results == [action, "brand-409"], results
    assert _active(pg, rid) == []


@pytest.mark.parametrize("action", ["rerun", "range"])
def test_brand_fix_first_then_the_competitor_is_refused(pg, tmp_path, monkeypatch, action):
    (pid, rid, _d, _rs), params = _setup(pg, tmp_path, monkeypatch)
    results: list = []
    _brand_admission(pg, pid, params, results)
    _hold_lock_then(pg, rid, results, action=action, hold_sec=0)
    assert results == ["brand", f"{action}-409"], results
    assert len(_active(pg, rid)) == 1


def test_crash_between_admission_and_enqueue_leaves_one_row(pg, tmp_path, monkeypatch):
    from app import tasks as tasks_mod

    (pid, rid, _d, _rs), params = _setup(pg, tmp_path, monkeypatch)

    def boom(_pid):
        raise ConnectionError("redis down")

    monkeypatch.setattr(tasks_mod, "enqueue_process_audio_patch", boom)
    results: list = []
    _brand_admission(pg, pid, params, results)
    assert results == ["brand"]
    assert len(_active(pg, rid)) == 1
    # The same admission replayed (the core re-plans with the SAME effect key and
    # finds it; a second execute is refused by the active row): still one row.
    _brand_admission(pg, pid, params, results)
    assert results == ["brand", "brand-409"]
    assert len(_active(pg, rid)) == 1
