"""AICP-389: the brand check + free fix inside qa:L, on the real autopilot core.

Driven from the AICP-366 harness (tests/test_autopilot_localisation_finish.py:
project 46cc6443's inputs — its es script says "Se llama Listenify, …" — real
small media, the real core, the real rerun/restore endpoints). Faked: the
local word ASR (no model in tests) and, per test, the detector's verdict
(``app.brand_check.check``) — the detector itself is covered on real audio by
tests/test_brand_check.py. The renders, hashes, previews, the apply handler,
the RunAudioPatch row and process_audio_patch (with a fake Drive) are real.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app import brand_check as bc
from app import media
from app import pipeline_audio_patch as pap
from app import tasks as tasks_mod
from app.autopilot.flows import localisation_brandfix as bf
from app.config import settings
from app.models import Run, RunAudioPatch, RunCaption
from app.models_autopilot import AutopilotApproval, AutopilotEffect

from tests.test_autopilot_localisation_finish import (  # noqa: F401 — fixtures
    FFMPEG, MALE, drive_to_rest, effects, fenv, go, make_project, rs_by_number,
)
from tests.test_autopilot_localisation_flow import (  # noqa: F401 — fixtures
    SF, answer, ap_row, engine, env, pending, q, settle, steps,
)

pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


class FakeDrive:
    def __init__(self):
        self.updates = []

    def update_file_content(self, file_id, local_path, mime_type="video/mp4"):
        self.updates.append((file_id, sha(local_path)))
        return {"id": file_id}

    def upload_file(self, *a, **k):  # pragma: no cover — never on this path
        raise AssertionError("a brand fix never uploads a new Drive file")


class FakeASR:
    def __init__(self, env):
        self.env = env

    def transcribe(self, path, *, start_sec, end_sec, language=None, timeout_sec=120):
        self.env.asr_calls.append((start_sec, end_sec, language))
        if self.env.asr_error:
            raise RuntimeError(self.env.asr_error)
        return SimpleNamespace(words=(), max_rss_kib=1234)


@pytest.fixture()
def benv(fenv, monkeypatch):
    env = fenv
    monkeypatch.setattr(settings, "AUTOPILOT_BRAND_FIX_ENABLED", True)
    env.asr_calls, env.asr_error, env.found = [], None, True
    env.patches, env.drive = [], FakeDrive()
    monkeypatch.setattr(bf, "_local_asr", lambda: FakeASR(env))
    monkeypatch.setattr(bf, "_audio_duration", lambda p: media.probe(p).duration_sec)
    real_check = bc.check

    def fake_check(samples, words, expected, brands, language, **kw):
        res = real_check(samples, words, expected, brands, language, **kw)
        if env.found:
            for r in res["occurrences"]:
                if r["brand"] == "Listenify" and r["verdict"] != "not_checked":
                    r.update(verdict="found", window=[0.8, 1.0], reason="seeded",
                             measurements={"fric_sec": 0.12, "pause_sec": 0.2})
                    break
        res["found"] = [r["occ_id"] for r in res["occurrences"] if r.get("verdict") == "found"]
        return res

    monkeypatch.setattr(bc, "check", fake_check)
    monkeypatch.setattr(tasks_mod, "enqueue_process_audio_patch", env.patches.append)

    @contextmanager
    def get_session():
        db = env.SF()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    monkeypatch.setattr(pap, "get_session", get_session)
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "")
    return env


def brandfix(env, prj):
    return [a for a in pending(env, prj, "handoff") if a.name.startswith("brandfix:")]


def run_patches(env):
    while env.patches:
        pap.process_audio_patch(env.patches.pop(0), gdrive=env.drive)


def set_drive(env, rid, fid="drive-rest"):
    db = env.SF()
    db.get(Run, rid).result_gdrive_file_id = fid
    db.commit()
    db.close()


def caption_effects(env, prj):
    return [e for e in effects(env, prj, "qa:es") if e.name.startswith("asr:")]


def test_found_ending_is_asked_before_the_asr_then_applied_in_place(benv):
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    set_drive(env, rest.id)
    base_path = q(env, Run, id=rest.id)[0].result_local_path
    base_sha = sha(base_path)

    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    (ask,) = brandfix(env, prj)
    assert ask.step_id == "qa:es" and ask.name.startswith(f"brandfix:{base_sha[:12]}:o")
    assert caption_effects(env, prj) == []  # the detector runs BEFORE the paid caption
    assert [o["id"] for o in ask.options] == ["apply", "keep"]
    apply = ask.options[0]["action"]
    p = apply["params"]
    assert apply["handler"] == bf.H_APPLY and ask.options[0]["cost_mcredits"] == 0
    assert p["base_sha256"] == base_sha and p["base_path"] == base_path
    assert p["candidate_sha256"] == sha(p["candidate_path"]) and p["window"] == [0.8, 1.0]
    assert p["filter_params"]["lowpass_hz"] == 2200 and p["filter_params"]["ramp_sec"] == 0.025
    assert [a["kind"] for a in ask.artifacts] == ["video", "video"]  # before / after previews
    assert ask.proposal["candidate_sha256"] == p["candidate_sha256"]
    assert "in the approved budget" in ask.proposal["cost"]  # the first caption is budgeted
    assert set(ask.bound_versions) >= {f"run:{rest.id}", f"project_transcript:{prj.pid}"}
    assert any(k.startswith("effect:") for k in ask.bound_versions)
    # the candidate: picture copied, same length
    a, b = bc.probe_streams(base_path), bc.probe_streams(p["candidate_path"])
    assert a["video_packets"] == b["video_packets"]

    answer(env, prj, ask, option="apply")
    settle(env, prj, until=lambda: bool(env.patches))
    (row,) = q(env, RunAudioPatch, run_id=rest.id)
    assert row.kind == "same_run_lowpass" and row.donor_run_id is None
    assert row.candidate_sha256 == p["candidate_sha256"] and row.base_sha256 == base_sha
    run_patches(env)
    row = q(env, RunAudioPatch, run_id=rest.id)[0]
    run = q(env, Run, id=rest.id)[0]
    assert row.status.value == "done"
    assert run.result_local_path == row.output_path != base_path
    assert sha(run.result_local_path) == p["candidate_sha256"]  # the approved bytes, promoted
    assert env.drive.updates == [("drive-rest", p["candidate_sha256"])]  # same Drive id
    assert run.result_gdrive_file_id == "drive-rest"

    # The next round runs on the patched final: the same window is never
    # offered again; then the (budgeted) caption and the listening.
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    assert brandfix(env, prj) == []
    checks = [e for e in effects(env, prj, "qa:es") if e.name.startswith("brand:")]
    assert len(checks) == 2 and checks[-1].result["final"] == run.result_local_path
    assert [r["verdict"] for r in checks[-1].result["occurrences"] if r["brand"] == "Listenify"][0] \
        == "patched"
    (cap,) = caption_effects(env, prj)
    assert cap.name.startswith("asr:r1:") and cap.params["run_id"] == rest.id
    listen = pending(env, prj, "listening_final")[0]
    assert listen.proposal["brand_check"]["status"] == "checked"
    assert ap_row(env, prj).status == "active"


def test_keep_is_not_asked_again_and_the_round_goes_on(benv):
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    ask = brandfix(env, prj)[0]
    answer(env, prj, ask, option="keep")
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    assert brandfix(env, prj) == []
    assert q(env, RunAudioPatch, run_id=rest.id) == []
    names = [a.name for a in q(env, AutopilotApproval, autopilot_id=prj.apid)
             if a.name.startswith("brandfix:")]
    assert names == [ask.name]
    assert pending(env, prj, "listening_final")[0].proposal["brand_check"]["kept"]


def test_a_changed_base_supersedes_the_ask_and_the_handler_refuses_stale_bytes(benv):
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    ask = brandfix(env, prj)[0]
    params = ask.options[0]["action"]["params"]
    # Someone rewrites the delivered final (same path, other bytes).
    with open(params["base_path"], "ab") as fh:
        fh.write(b"\0")
    db = env.SF()
    from app.models import VideoProject

    project = db.get(VideoProject, prj.pid)
    h = bf.LocalisationApplyBrandAudioPatch()
    from app.autopilot.contract import ActionRefused

    eff = SimpleNamespace(created_refs={}, result={})
    with pytest.raises(ActionRefused) as exc:
        h.execute(db, project, h.validate(db, project, params), eff)
    assert exc.value.status == 409 and "stale" in exc.value.detail
    db.rollback()
    db.close()
    assert q(env, RunAudioPatch, run_id=rest.id) == []
    settle(env, prj, rounds=3)
    old = q(env, AutopilotApproval, id=ask.id)[0]
    assert old.status == "superseded"


def test_promotion_refuses_a_base_changed_after_admission(benv):
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    first = brandfix(env, prj)[0]
    answer(env, prj, first, option="apply")
    settle(env, prj, until=lambda: bool(env.patches))
    run = q(env, Run, id=rest.id)[0]
    before = sha(run.result_local_path)
    with open(run.result_local_path, "ab") as fh:  # bytes move, path stays
        fh.write(b"\0")
    with pytest.raises(RuntimeError, match="bytes changed"):
        run_patches(env)
    row = q(env, RunAudioPatch, run_id=rest.id)[0]
    assert row.status.value == "failed" and env.drive.updates == []
    assert q(env, Run, id=rest.id)[0].result_local_path == run.result_local_path
    assert before != sha(run.result_local_path)
    # Never "fixed": the moved base is a new final, checked (and asked) again.
    go(env, prj, until=lambda: any(a.name != first.name for a in brandfix(env, prj)))
    again = brandfix(env, prj)[0]
    assert again.options[0]["action"]["params"]["base_sha256"] == sha(run.result_local_path)


def test_unavailable_check_is_retried_then_handed_off(benv):
    env = benv
    env.asr_error = "model manifest integrity failure"
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    go(env, prj, until=lambda: any(a.name.startswith("brandcheck:") for a in pending(env, prj, "handoff")))
    checks = [e for e in effects(env, prj, "qa:es") if e.name.startswith("brand:")]
    assert len(checks) == bf.MAX_CHECK_ATTEMPTS and all(e.state == "failed" for e in checks)
    assert caption_effects(env, prj) == []
    a = [a for a in pending(env, prj, "handoff") if a.name.startswith("brandcheck:")][0]
    assert [o["id"] for o in a.options] == ["continue", "stop"]
    answer(env, prj, a, option="continue")
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    assert pending(env, prj, "listening_final")[0].proposal["brand_check"]["status"] == "unavailable"


def test_a_regeneration_offered_after_the_fix_says_it_drops_it(benv):
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj, hz={3: MALE})  # + a gender error in segment 3
    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    assert pending(env, prj, "regeneration") == []  # the brand ask comes first
    answer(env, prj, brandfix(env, prj)[0], option="apply")
    settle(env, prj, until=lambda: bool(env.patches))
    run_patches(env)
    go(env, prj, until=lambda: any(a.name.startswith("regen:") for a in pending(env, prj, "regeneration")))
    regen = [a for a in pending(env, prj, "regeneration") if a.name.startswith("regen:")][0]
    rerun = next(o for o in regen.options if o["id"] == "rerun")
    assert rerun["action"]["params"]["discard_audio_patch"] is True
    assert "drops the brand fix" in rerun["consequence"]
    assert "drops the brand fix" in regen.proposal["drops_audio_fix"]


def test_disabled_flag_runs_no_check(benv, monkeypatch):
    env = benv
    monkeypatch.setattr(settings, "AUTOPILOT_BRAND_FIX_ENABLED", False)
    prj = make_project(env)
    drive_to_rest(env, prj)
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    assert not [e for e in effects(env, prj, "qa:es") if e.name.startswith("brand:")]
    assert env.asr_calls == []


def test_compute_runs_on_the_brand_asr_queue():
    from app.autopilot import registry

    assert registry.compute_queue(bf.C_BRAND) == "brand_asr"
    assert registry.compute_queue("localisation_segment_qa") == "autopilot_compute"


def test_read_paths_tolerate_a_filter_row(benv):
    """The patch list API and the run page's fragment show a same-run fix."""
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    answer(env, prj, brandfix(env, prj)[0], option="apply")
    settle(env, prj, until=lambda: bool(env.patches))
    run_patches(env)
    import app.api_v2 as api_v2
    from starlette.requests import Request

    from app import web_v2

    db = env.SF()
    try:
        (item,) = api_v2.list_run_audio_patches(rest.id, db=db)
        assert item.kind == "same_run_lowpass" and item.donor_run_id is None
        assert item.donor_label.startswith("this run")
        req = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""})
        html = web_v2.run_audio_patches_fragment(rest.id, req, db=db).body.decode()
        assert "brand fix: low-pass 2200 Hz" in html
    finally:
        db.close()


def test_gate_promote_refuses_a_reaped_patch_before_drive(benv):
    """Gate R1: the patch row is locked and must still be `delivering`."""
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    set_drive(env, rest.id)
    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    answer(env, prj, brandfix(env, prj)[0], option="apply")
    settle(env, prj, until=lambda: bool(env.patches))
    pid = env.patches.pop()
    from app.state_machine import AudioPatchStatus

    db = env.SF()
    patch = db.get(RunAudioPatch, pid)
    run = db.get(Run, patch.run_id)
    patch.output_path = patch.candidate_path
    patch.status = AudioPatchStatus.delivering
    db.commit()
    other = env.SF()
    other.get(RunAudioPatch, pid).status = AudioPatchStatus.failed  # the reaper wins first
    other.commit()
    other.close()
    with pytest.raises(RuntimeError, match="not delivering"):
        pap._promote(db, run, patch, gdrive=env.drive, filename="x.mp4")
    db.rollback()
    db.close()
    assert env.drive.updates == []


def test_gate_a_check_of_another_transcript_is_not_usable(tmp_path):
    """Gate R2: a result is evidence only for the transcript it was computed on
    (a transcript edit in the flow reopens the translation anyway; this guards
    the window between)."""
    from app.autopilot.flows import localisation_qa as lqa

    final = tmp_path / "f.mp4"
    final.write_bytes(b"x" * 10)
    run = SimpleNamespace(result_local_path=str(final))
    res = {"status": "ok", "stable": True, "final": str(final), "base_sig": lqa.file_sig(str(final)),
           "detector_version": bc.DETECTOR_VERSION, "transcript_version": "tv1"}
    assert bf._result_usable(res, run, "tv1")
    assert not bf._result_usable(res, run, "tv2")
    final.write_bytes(b"y" * 11)
    assert not bf._result_usable(res, run, "tv1")


def test_gate_skip_windows_follow_the_current_lineage_only(benv):
    """Gate R3: a patch dropped by a rerun no longer marks its window as fixed."""
    env = benv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj)
    go(env, prj, until=lambda: bool(brandfix(env, prj)))
    answer(env, prj, brandfix(env, prj)[0], option="apply")
    settle(env, prj, until=lambda: bool(env.patches))
    run_patches(env)
    db = env.SF()
    try:
        assert [p.kind for p in bf.lineage_patches(db, rest.id)] == ["same_run_lowpass"]
        run = db.get(Run, rest.id)
        # a rerun re-stitched the final: the patch is not in its lineage any more
        import shutil

        fresh = str(env.tmp / "restitched.mp4")
        shutil.copyfile(q(env, RunAudioPatch, run_id=rest.id)[0].base_result_path, fresh)
        run.result_local_path = fresh
        db.commit()
        assert bf.lineage_patches(db, rest.id) == []
        assert bf.current_patch(db, rest.id) is None
    finally:
        db.close()


def test_gate_check_attempts_are_capped_even_when_the_script_keeps_moving(monkeypatch):
    """Gate 2 R1: at most MAX_CHECK_ATTEMPTS + 3 checks per round, then the handoff."""
    from datetime import datetime, timezone

    from app.autopilot.contract import Ask, Compute

    tag = "r1:abc"
    effs = []
    t0 = datetime(2026, 11, 5, tzinfo=timezone.utc)

    def mk(i):
        return SimpleNamespace(name=f"brand:{tag}" + (f":r{i + 1}" if i else ""), state="succeeded",
                               created_at=t0.replace(minute=i), error=None, id=f"e{i}",
                               params={"args": {}},
                               result={"status": "ok", "transcript_version": f"old{i}"})

    class Ctx:
        db = None
        now = t0
        step_id, step_generation = "qa:es", 1
        project = SimpleNamespace(id="p", transcript={"brands": [{"name": "Listenify"}]})

        def transcript_version(self):
            return "current"

        def run_version(self, rid):
            return "v"

        def effects(self, prefix=""):
            return [e for e in effs if e.name.startswith(prefix)]

        def approval(self, name):
            return None

        def effect(self, name):
            return next((e for e in effs if e.name == name), None)

    monkeypatch.setattr(settings, "AUTOPILOT_BRAND_FIX_ENABLED", True)
    monkeypatch.setattr(bf, "lineage_patches", lambda db, rid: [])
    monkeypatch.setattr(bf, "run_patches", lambda db, rid: [])
    monkeypatch.setattr(bf.lqa, "final_script", lambda pieces, secs: [
        {"start": 0, "end": 2, "text": "Es Listenify.", "tag": "A", "pos": 0}])
    run = SimpleNamespace(id="r", short_id="r", result_local_path="/nope")
    kinds = []
    for i in range(12):
        d, _ = bf.brand_round(Ctx(), "es", run, tag, 1, [], None, lambda *n: list(n))
        kinds.append(type(d).__name__)
        if isinstance(d, Compute):
            effs.append(mk(len(effs)))
        else:
            assert isinstance(d, Ask) and d.name == f"brandcheck:{tag}"
            break
    assert kinds.count("Compute") == bf.MAX_CHECK_ATTEMPTS + 3 and kinds[-1] == "Ask"
