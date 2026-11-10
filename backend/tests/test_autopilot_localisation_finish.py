"""AICP-366 acceptance: the localisation finishing steps on the real autopilot core.

Driven from C1's harness (tests/test_autopilot_localisation_flow.py: project
46cc6443's inputs, mocked transcription/translation, the real core, durable
jobs, create_run / rerun_segments / restore endpoints) with REAL small media:
every generated clip is an ffmpeg tone of its window's length (220 Hz = the
female voices of 46cc6443, 110 Hz = the seeded gender error), so the segment
QA compute (lengths, sheets, caption detector, drift, voice pitch against the
accepted probe) runs for real. The caption worker (the ASR is the caption
job's transcription) and the tail worker are faked at their enqueue.

Acceptance (AICP-366): a stitched final with a seeded gender error in segment
3 produces a regeneration approval for segment 3 only, with the cue and the
cost; after approve + a mocked rerun the listening_final approval is created;
captions and the CTA tail are produced without paid calls; the TEMP clip is
deleted; the language's spend is journaled.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import pytest
from sqlalchemy import select

import app.api_v2 as api_v2_module
from app import media
from app.autopilot import core
from app.autopilot.flows import localisation as flow_mod
from app.autopilot.flows import localisation_finish as fin
from app.autopilot.flows import localisation_finish_handlers as fh
from app.autopilot.flows import localisation_qa as lqa
from app.models import (
    KieSpendReservation, LibraryClip, Run, RunCaption, RunSegment, RunSegmentTake, RunTailEdit,
    SegmentDef, VideoProject,
)
from app.models_autopilot import AutopilotEffect, AutopilotJournal, ProjectAutopilot
from app.state_machine import CaptionStatus, RunStatus, SegmentStatus, TailEditStatus

from tests.test_autopilot_localisation_flow import (  # noqa: F401 — fixtures
    DURATION, SF, answer, ap_row, drive_to_budget, engine, env, pending, q, runs, settle, steps, tick,
)
from tests.test_autopilot_localisation_plan import transcript_v3

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
REAL_PROBE = media.probe
FEMALE, MALE = 220, 110


def tone(path, seconds, hz):
    subprocess.run([
        FFMPEG, "-nostdin", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=0x405060:s=96x160:r=12:d={seconds:.3f}",
        "-f", "lavfi", "-i", f"sine=frequency={hz}:sample_rate=16000:duration={seconds:.3f}",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", str(path),
    ], check=True)
    return str(path)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture()
def fenv(env, monkeypatch):
    """C1's env with the real ffprobe and faked caption / tail workers."""
    monkeypatch.setattr(media, "probe", REAL_PROBE)
    env.captions, env.tails = [], []
    monkeypatch.setattr(api_v2_module, "enqueue_process_caption", env.captions.append)
    monkeypatch.setattr(api_v2_module, "enqueue_process_tail_edit", env.tails.append)
    monkeypatch.setattr(flow_mod, "_clip_lookup", lambda cid: _lookup(env, cid))
    env.asr = None  # optional (lines) -> lines mutation of the fake ASR
    env.cache = {}
    return env


def _lookup(env, cid):
    db = env.SF()
    try:
        c = db.get(LibraryClip, cid)
        return None if c is None else {"id": c.id, "language": c.language}
    finally:
        db.close()


def media_file(env, seconds, hz):
    key = (round(seconds, 3), hz)
    if key not in env.cache:
        env.cache[key] = tone(env.tmp / f"tone-{key[0]}-{hz}.mp4", seconds, hz)
    dst = env.tmp / f"m-{uuid.uuid4().hex[:8]}.mp4"
    shutil.copyfile(env.cache[key], dst)
    return str(dst)


def make_project(env, *, tails=None):
    db = env.SF()
    src = media_file(env, DURATION, FEMALE)
    p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=src,
                     status="ready", project_type="localisation", mode="assistant",
                     localisation_scope="whole", duration_sec=DURATION, mute_source=True,
                     transcript=transcript_v3(), transcript_status="ready")
    db.add(p)
    db.flush()
    from tests.test_autopilot_localisation_plan import SEGMENTS

    for s in SEGMENTS:
        db.add(SegmentDef(project_id=p.id, index=s["index"], start_sec=s["start_sec"],
                          end_sec=s["end_sec"], action="swap", pre_roll_sec=0.0, post_roll_sec=0.0))
    cta = LibraryClip(id=str(uuid.uuid4()), name="CTA es", local_path=media_file(env, 3.0, FEMALE),
                      duration_sec=3.0, width=96, height=160, fps=12.0, language="es")
    db.add(cta)
    db.commit()  # validate_params looks the tail clip up in its own session
    params = {"languages": ["es"]}
    if tails is not False:
        params["tails"] = {"es": cta.id}
    params = flow_mod.FLOW.validate_params(p, params)
    ap = core.create_autopilot(db, p, flow_mod.FLOW, params)
    db.commit()
    from types import SimpleNamespace

    out = SimpleNamespace(pid=p.id, apid=ap.id, cta=cta.id, src=src)
    db.close()
    return out


def _windows(db, project):
    defs = db.execute(select(SegmentDef).where(SegmentDef.project_id == project.id)
                      .order_by(SegmentDef.index)).scalars().all()
    return defs


def finish(env, rid, *, hz=None, status=RunStatus.done):
    """The run pipeline, faked with real files: every pending segment gets a tone clip of
    its window's length (hz per segment number), a new final, the run is done."""
    hz = hz or {}
    db = env.SF()
    run = db.get(Run, rid)
    project = db.get(VideoProject, run.project_id)
    defs = _windows(db, project)
    number = {d.id: i + 1 for i, d in enumerate(defs)}
    plan = run.stitch_plan or []
    active = {e["segment_def_id"] for e in plan if e.get("kind") == "segment"}
    existing = {rs.segment_def_id: rs for rs in db.execute(
        select(RunSegment).where(RunSegment.run_id == rid)).scalars()}
    for sd in defs:
        if sd.id not in active:
            continue
        rs = existing.get(sd.id)
        if rs is None:
            rs = RunSegment(run_id=rid, segment_def_id=sd.id, index=sd.index)
            db.add(rs)
        if rs.status == SegmentStatus.completed and rs.local_result_path:
            continue  # untouched by a rerun
        rs.status = SegmentStatus.completed
        rs.local_result_path = media_file(env, sd.end_sec - sd.start_sec, hz.get(number[sd.id], FEMALE))
        rs.seedance_task_id = f"t-{uuid.uuid4().hex[:6]}"
    db.flush()
    if len(active) == 1:  # a probe: its final IS its one generated clip
        clip = db.execute(select(RunSegment.local_result_path).where(
            RunSegment.run_id == rid)).scalars().first()
        dst = env.tmp / f"final-{uuid.uuid4().hex[:8]}.mp4"
        shutil.copyfile(clip, dst)
        run.result_local_path = str(dst)
    else:
        run.result_local_path = media_file(env, 2.0 + len(env.cache) * 0.01, FEMALE)
    run.segment_snapshot = [{"id": d.id, "index": d.index, "action": d.action, "start_sec": d.start_sec,
                             "end_sec": d.end_sec, "pre_roll_sec": 0.0, "post_roll_sec": 0.0}
                            for d in defs]
    run.status = status
    db.commit()
    db.close()


def expected_lines(db, run):
    """What the ASR of a perfect take hears: every scripted line, in stitch order."""
    out, t = [], 0.0
    for e in run.stitch_plan or []:
        if e.get("kind") == "library":
            clip = db.get(LibraryClip, e["clip_id"])
            # the TEMP probe clip: the probe run's own prompt
            probe = db.execute(select(Run).where(Run.project_id == run.project_id,
                                                 Run.id != run.id)).scalars().first()
            prompt = db.execute(select(RunSegment.prompt_override).where(
                RunSegment.run_id == probe.id)).scalar_one()
            length = clip.duration_sec or 12.0
        else:
            rs = db.execute(select(RunSegment).where(
                RunSegment.run_id == run.id,
                RunSegment.segment_def_id == e["segment_def_id"])).scalar_one()
            sd = db.get(SegmentDef, e["segment_def_id"])
            prompt, length = rs.prompt_override or run.prompt, sd.end_sec - sd.start_sec
        for ln in lqa.parse_script(prompt):
            if not ln.background:
                out.append({"start": round(t + ln.start, 2), "end": round(t + ln.end, 2),
                            "text": ln.text})
        t += length
    return out


def run_captions(env):
    """The caption worker: ASR (unless the lines are manual) + burn + deliver."""
    while env.captions:
        cid = env.captions.pop(0)
        db = env.SF()
        cap = db.get(RunCaption, cid)
        run = db.get(Run, cap.run_id)
        if cap.lines_source == "manual" and getattr(env, "fail_burn_once", False):
            env.fail_burn_once = False
            cap.status, cap.error_message = CaptionStatus.failed, "Drive upload failed"
            db.commit()
            db.close()
            continue
        if cap.lines_source != "manual":
            lines = expected_lines(db, run)
            cap.lines = env.asr(lines) if env.asr else lines
            cap.lines_source = "transcribe"
        cap.audio_sha256 = media.audio_fingerprint(run.result_local_path)
        out = env.tmp / f"cap-{cid[:8]}-{uuid.uuid4().hex[:4]}.mp4"
        shutil.copyfile(run.result_local_path, out)
        cap.output_path, cap.status = str(out), CaptionStatus.done
        db.commit()
        db.close()


def run_tails(env):
    while env.tails:
        tid = env.tails.pop(0)
        db = env.SF()
        edit = db.get(RunTailEdit, tid)
        cap = db.get(RunCaption, edit.source_caption_id)
        out = env.tmp / f"tail-{tid[:8]}.mp4"
        shutil.copyfile(cap.output_path, out)
        edit.output_path, edit.status = str(out), TailEditStatus.done
        db.commit()
        db.close()


def go(env, prj, until, rounds=40):
    def step():
        run_captions(env)
        run_tails(env)
        return until()
    try:
        settle(env, prj, rounds=rounds, until=step)
    except AssertionError:
        raise AssertionError("stalled:\n" + dump(env, prj)) from None


def dump(env, prj) -> str:
    db = env.SF()
    try:
        st = [f"{s.step_id}: {s.status} — {s.reason}" for s in db.execute(select(
            __import__("app.models_autopilot", fromlist=["AutopilotStep"]).AutopilotStep).where(
            __import__("app.models_autopilot", fromlist=["AutopilotStep"]).AutopilotStep.autopilot_id
            == prj.apid)).scalars()]
        js = db.execute(select(AutopilotJournal).where(AutopilotJournal.autopilot_id == prj.apid)
                        .order_by(AutopilotJournal.id.desc()).limit(25)).scalars().all()
        ap = db.get(ProjectAutopilot, prj.apid)
        qa = []
        for e in db.execute(select(AutopilotEffect).where(
                AutopilotEffect.autopilot_id == prj.apid, AutopilotEffect.name.like("segqa:%"))).scalars():
            r = e.result or {}
            qa.append(f"{e.name} {e.state} {e.error} stable={r.get('stable')} "
                      f"baseline={r.get('baseline')} files_ok={fin._files_current(r) if r else None}")
            for pc in r.get("pieces") or []:
                qa.append(f"  pos{pc['pos']} seg{pc.get('seg_number')} {pc['kind']} sec={pc.get('clip_sec')} "
                          f"issues={pc.get('issues')} warn={[w.get('kind') for w in pc.get('warnings') or []]} "
                          f"voices={pc.get('voices')}")
        return "\n".join([f"autopilot {ap.status}: {ap.status_reason}"] + st + qa + ["--- journal"]
                         + [f"{j.kind} {j.step_id}: {j.message}" for j in reversed(js)])
    finally:
        db.close()


def drive_to_rest(env, prj, *, hz=None):
    """C1 with real media: budget -> probe (220 Hz) -> listening -> the rest run done."""
    answer(env, prj, drive_to_budget(env, prj))
    settle(env, prj, until=lambda: len(runs(env, prj)) == 1)
    probe = runs(env, prj)[0]
    finish(env, probe.id)
    settle(env, prj, until=lambda: bool(pending(env, prj, "listening_probe")))
    answer(env, prj, pending(env, prj, "listening_probe")[0])
    settle(env, prj, until=lambda: len(runs(env, prj)) == 2)
    rest = runs(env, prj)[1]
    finish(env, rest.id, hz=hz)
    return probe, rest


def effects(env, prj, step=None):
    rows = q(env, AutopilotEffect, autopilot_id=prj.apid)
    return [e for e in rows if step is None or e.step_id == step]


def rs_by_number(env, rid):
    db = env.SF()
    try:
        run = db.get(Run, rid)
        defs = _windows(db, db.get(VideoProject, run.project_id))
        number = {d.id: i + 1 for i, d in enumerate(defs)}
        return {number[r.segment_def_id]: r for r in db.execute(
            select(RunSegment).where(RunSegment.run_id == rid)).scalars()}
    finally:
        db.close()


def reservations(env):
    db = env.SF()
    try:
        return db.execute(select(KieSpendReservation)).scalars().all()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def test_seeded_gender_error_in_segment_3(fenv):
    env = fenv
    prj = make_project(env)
    probe, rest = drive_to_rest(env, prj, hz={3: MALE})

    go(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    regen = pending(env, prj, "regeneration")[0]
    assert regen.step_id == "qa:es" and regen.name.startswith("regen:r1:")
    rerun = next(o for o in regen.options if o["id"] == "rerun")
    segs = rerun["action"]["params"]["segments"]
    by_number = rs_by_number(env, rest.id)
    assert [s["run_segment_id"] for s in segs] == [by_number[3].id]  # segment 3 only
    assert "INTERVIEWER is a female voice in every line of this clip — never a male voice." in \
        segs[0]["prompt"] or "WOMAN is a female voice" in segs[0]["prompt"]
    assert lqa.QA_CUE_HEADER in segs[0]["prompt"]
    assert rerun["cost_mcredits"] > 0  # the core's estimate of the rerun
    assert list(regen.proposal["issues"]) == ["3"]
    assert regen.proposal["issues"]["3"][0]["kind"] == "gender"
    assert regen.proposal["issues"]["3"][0]["measured"] == "male"
    assert [a["kind"] for a in regen.artifacts] == ["video", "image"]
    assert pending(env, prj, "listening_final") == []
    budget_before = ap_row(env, prj).approved_budget_mcredits

    answer(env, prj, regen, option="rerun")
    settle(env, prj, until=lambda: q(env, Run, id=rest.id)[0].status != RunStatus.done)
    assert ap_row(env, prj).approved_budget_mcredits > budget_before  # Q6
    db = env.SF()
    assert db.get(RunSegment, by_number[3].id).prompt_override == segs[0]["prompt"]
    assert db.execute(select(RunSegmentTake).where(
        RunSegmentTake.run_segment_id == by_number[3].id)).scalars().first() is not None  # AICP-342
    db.close()
    finish(env, rest.id)  # the mocked rerun: segment 3 now in a female voice

    # Round 2: its ASR re-check is a regeneration of its own (design §5b.1).
    go(env, prj, until=lambda: bool(pending(env, prj, "regeneration")
                                    or pending(env, prj, "listening_final")))
    asr = pending(env, prj, "regeneration")[0]
    assert asr.name.startswith("asr:r2:") and asr.name.endswith("#regeneration")
    # The deliverable round's ASR mishears one word (a QA pass: one word of a line).
    env.asr = lambda lines: [dict(ln, text=ln["text"].replace("favorita", "faborita")) for ln in lines]
    answer(env, prj, asr)
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    listen = pending(env, prj, "listening_final")[0]
    assert listen.subject == "es" and listen.name.startswith("listen:r2:")
    assert listen.artifacts[0]["kind"] == "video" and len(listen.artifacts) == 7  # final + 6 sheets
    assert listen.proposal["issues"] == {}

    paid_before = {r.scope_key for r in reservations(env)}
    env.fail_burn_once = True  # the first burn of the fixed lines fails: retried as is
    answer(env, prj, listen)
    go(env, prj, until=lambda: ap_row(env, prj).status == "completed")
    st = steps(env, prj)
    assert st["qa:es"].output["round"] == 2
    # Captions and tail: free, no new paid reservation.
    for e in effects(env, prj, "captions:es") + effects(env, prj, "tail:es") + effects(env, prj, "cleanup:es"):
        assert e.paid is False
    assert {r.scope_key for r in reservations(env)} == paid_before
    # The tail: off the caption, cut at duration - 0.01 floored, delivered under 99 MB.
    tail = st["tail:es"].output
    db = env.SF()
    edit = db.get(RunTailEdit, tail["tail_edit_id"])
    cap = db.get(RunCaption, tail["caption_id"])
    assert edit.source_type == "caption" and edit.source_caption_id == cap.id
    assert edit.library_clip_id == prj.cta
    assert edit.cut_sec == lqa.tail_cut_sec(REAL_PROBE(cap.output_path).duration_sec)
    assert tail["size_bytes"] <= lqa.DELIVERABLE_MAX_BYTES and len(tail["sha256"]) == 64
    # Captions: the misheard word was fixed back to the script, burn only.
    assert cap.lines_source == "manual"
    assert not any("faborita" in ln["text"] for ln in cap.lines)
    assert any("favorita" in ln["text"] for ln in cap.lines)
    assert st["captions:es"].output["mode"] == "asr_fix"
    fixes = sorted((e for e in effects(env, prj, "captions:es")), key=lambda e: e.created_at)
    assert [e.name for e in fixes] == ["fix", "fix:r2"]
    assert fixes[1].params["lines"] == fixes[0].params["lines"] == cap.lines  # not the script
    # Cleanup: the TEMP clip is gone, the spend is journaled.
    temp = st["rest:es"].output["temp_clip_id"]
    assert db.get(LibraryClip, temp) is None
    assert not os.path.exists(fh.temp_clip_path(temp))
    spend = st["cleanup:es"].output["spend"]
    assert spend["language"] == "es" and spend["credits"] > 0
    journal = db.execute(select(AutopilotJournal).where(
        AutopilotJournal.autopilot_id == prj.apid, AutopilotJournal.step_id == "cleanup:es")).scalars().all()
    assert any("es spend:" in (j.message or "") for j in journal)
    # The ASR captions were reserved under their own scope (the caption:None fix).
    scopes = {r.scope_key for r in db.execute(select(KieSpendReservation)).scalars()}
    assert "caption:None" not in scopes
    db.close()
    # A rerun after the cleanup is refused before anything is billed.
    db = env.SF()
    from fastapi import HTTPException
    from app.schemas_v2 import SegmentRerunItem, SegmentsRerunRequest

    with pytest.raises(HTTPException) as exc:
        api_v2_module.rerun_segments(rest.id, SegmentsRerunRequest(
            segments=[SegmentRerunItem(run_segment_id=by_number[2].id)]), db=db)
    assert exc.value.status_code == 409 and "no longer exist" in exc.value.detail
    db.close()
    assert {r.scope_key for r in reservations(env)} == paid_before


def test_missing_line_then_rejected_final_targets_the_named_segment(fenv):
    env = fenv
    prj = make_project(env, tails=False)
    probe, rest = drive_to_rest(env, prj)
    by_number = rs_by_number(env, rest.id)
    db = env.SF()
    seg4_first = lqa.parse_script(db.get(RunSegment, by_number[4].id).prompt_override)[0].text
    db.close()
    env.asr = lambda lines: [ln for ln in lines if ln["text"] != seg4_first]
    go(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    regen = pending(env, prj, "regeneration")[0]
    assert list(regen.proposal["issues"]) == ["4"]
    assert regen.proposal["issues"]["4"][0]["kind"] == "missing_line"
    rerun = next(o for o in regen.options if o["id"] == "rerun")
    assert [s["run_segment_id"] for s in rerun["action"]["params"]["segments"]] == [by_number[4].id]
    assert "none may be skipped" in rerun["action"]["params"]["segments"][0]["prompt"]

    answer(env, prj, regen, option="accept")  # keep it, go on to listening
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    listen = pending(env, prj, "listening_final")[0]
    answer(env, prj, listen, "reject", comment="segment 5: the interviewer sounds like a man")
    go(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    fix = pending(env, prj, "regeneration")[0]
    assert fix.name.startswith("fix:r1:")
    params = next(o for o in fix.options if o["id"] == "rerun")["action"]["params"]
    assert [s["run_segment_id"] for s in params["segments"]] == [by_number[5].id]
    prompt = params["segments"][0]["prompt"]
    assert prompt.count("- segment 5: the interviewer sounds like a man") == 1
    assert "never a male voice" in prompt
    # Both answers (the accepted proposal and the rejection) were consumed by the cue.
    from app.models_autopilot import AutopilotApproval

    rows = {a.name: a.status for a in q(env, AutopilotApproval, autopilot_id=prj.apid)}
    assert rows[regen.name] == "consumed" and rows[listen.name] == "consumed"
    env.hooks.now.advance(3600)
    tick(env, prj)
    assert ap_row(env, prj).status == "active"

    answer(env, prj, fix, option="rerun")
    settle(env, prj, until=lambda: q(env, Run, id=rest.id)[0].status != RunStatus.done)
    db = env.SF()
    assert db.get(RunSegment, by_number[5].id).prompt_override == prompt
    db.close()


def test_a_worse_rerun_restores_the_better_take(fenv):
    env = fenv
    prj = make_project(env, tails=False)
    probe, rest = drive_to_rest(env, prj, hz={3: MALE})
    go(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    answer(env, prj, pending(env, prj, "regeneration")[0], option="rerun")
    settle(env, prj, until=lambda: q(env, Run, id=rest.id)[0].status != RunStatus.done)
    db = env.SF()
    by_number = rs_by_number(env, rest.id)
    old_task = db.execute(select(RunSegmentTake).where(
        RunSegmentTake.run_segment_id == by_number[3].id)).scalars().one().seedance_task_id
    db.close()
    # The regenerated segment 3 is male AGAIN and 1.5 s too long: two issues > one.
    db = env.SF()
    rs = db.get(RunSegment, by_number[3].id)
    sd = db.get(SegmentDef, rs.segment_def_id)
    rs.status, rs.local_result_path = SegmentStatus.completed, media_file(env, sd.end_sec - sd.start_sec + 1.5, MALE)
    rs.seedance_task_id = "t-worse"
    run = db.get(Run, rest.id)
    run.result_local_path, run.status = media_file(env, 2.5, FEMALE), RunStatus.done
    db.commit()
    db.close()
    settle(env, prj, until=lambda: any(e.name == "best:rerun:r1" for e in effects(env, prj, "qa:es")))
    best = next(e for e in effects(env, prj, "qa:es") if e.name == "best:rerun:r1")
    db = env.SF()
    take = db.get(RunSegmentTake, best.params["takes"][0])
    assert take.seedance_task_id == old_task and best.handler == "restore_takes"
    # decided before that round's ASR: no ASR caption for the worse take
    assert not [e for e in effects(env, prj, "qa:es") if e.name.startswith("asr:r2:")]
    db.close()


def test_garbled_asr_gives_captions_from_the_script(fenv):
    env = fenv
    prj = make_project(env, tails=False)
    drive_to_rest(env, prj)
    env.asr = lambda lines: [dict(ln, text="zrbl kwak tsss") for ln in lines]
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    listen = pending(env, prj, "listening_final")[0]
    assert listen.proposal["asr"]["garbled"] is True and listen.proposal["issues"] == {}
    answer(env, prj, listen)
    go(env, prj, until=lambda: getattr(steps(env, prj).get("captions:es"), "status", None) == "completed")
    out = steps(env, prj)["captions:es"].output
    assert out["mode"] == "script"
    db = env.SF()
    cap = db.get(RunCaption, out["caption_id"])
    assert cap.lines_source == "manual" and not any("zrbl" in ln["text"] for ln in cap.lines)
    db.close()
    go(env, prj, until=lambda: ap_row(env, prj).status == "completed")
    assert steps(env, prj)["tail:es"].output["tail_edit_id"] is None  # no CTA tail for es


def _set_placement(env, prj, scan, derived_at=None):
    db = env.SF()
    p = db.get(VideoProject, prj.pid)
    p.caption_placement = {"mode": "band", "center_y": 0.62, "band": [0.1, 0.58, 0.9, 0.66],
                           "source": {"kind": "inherited", "scan": scan, "rev": 1,
                                      "derived_at": derived_at or env.hooks.now().isoformat()}}
    db.commit()
    db.close()


def test_pending_caption_placement_waits_before_the_asr(fenv):
    env = fenv
    prj = make_project(env, tails=False)
    drive_to_rest(env, prj)
    _set_placement(env, prj, "pending", derived_at=env.hooks.now().isoformat())
    for _ in range(4):
        settle(env, prj, rounds=1)
        run_captions(env)
    assert not [e for e in effects(env, prj, "qa:es") if e.name.startswith("asr:")]
    assert "caption placement" in (steps(env, prj)["qa:es"].reason or "")
    # The scan settles: the ASR caption freezes the settled placement.
    _set_placement(env, prj, "done")
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_final")))
    asr = next(e for e in effects(env, prj, "qa:es") if e.name.startswith("asr:"))
    db = env.SF()
    cap = db.get(RunCaption, asr.created_refs["caption"])
    assert cap.placement["source"]["scan"] == "done" and cap.placement["center_y"] == 0.62
    db.close()


def test_failed_caption_placement_is_derived_again(fenv, monkeypatch):
    from app import caption_placement as cp

    env = fenv
    prj = make_project(env, tails=False)
    drive_to_rest(env, prj)
    _set_placement(env, prj, "failed")
    calls = []

    def fake_request(db, project):
        db.refresh(project, with_for_update=True)
        project.caption_placement = dict(project.caption_placement,
                                         source=dict(project.caption_placement["source"],
                                                     scan="pending", rev=2))
        db.commit()
        calls.append(project.id)
        return 2

    monkeypatch.setattr(cp, "removal_chain", lambda db, project: [("run", "ancestor", None)])
    monkeypatch.setattr(cp, "request_derivation", fake_request)
    monkeypatch.setattr(cp, "enqueue_scan", lambda pid, rev: calls.append(("scan", rev)) or True)
    for _ in range(5):
        settle(env, prj, rounds=1)
    assert calls == [prj.pid, ("scan", 2)]
    eff = next(e for e in effects(env, prj, "qa:es") if e.name == "placement")
    assert eff.handler == "localisation_rederive_placement" and eff.result["rev"] == 2
    assert not [e for e in effects(env, prj, "qa:es") if e.name.startswith("asr:")]  # waits again


def test_failed_placement_without_lineage_captions_with_the_fallback(fenv, monkeypatch):
    from app import caption_placement as cp

    env = fenv
    prj = make_project(env, tails=False)
    drive_to_rest(env, prj)
    _set_placement(env, prj, "failed")
    monkeypatch.setattr(cp, "removal_chain", lambda db, project: [])  # the lineage was deleted
    go(env, prj, until=lambda: any(e.name.startswith("asr:") for e in effects(env, prj, "qa:es")))
    assert not [e for e in effects(env, prj, "qa:es") if e.name.startswith("placement")]


# ---------------------------------------------------------------------------
# Hooks and the endpoint fix
# ---------------------------------------------------------------------------


def test_caption_placement_hooks():
    import inspect

    assert "placement" in inspect.signature(api_v2_module.create_caption).parameters
    assert fh.caption_placement_kwargs() == {"placement": "project"}
    proj = VideoProject(project_type="localisation")
    assert fh.frozen_caption_placement(proj) == {"placement": None}
    assert fh.caption_placement_state(proj) == "none"
    proj.caption_placement = {"mode": "band", "source": {"scan": "done"}}
    frozen = fh.frozen_caption_placement(proj)["placement"]
    assert frozen == proj.caption_placement and frozen is not proj.caption_placement
    assert fh.caption_placement_state(proj) == "settled"
    proj.caption_placement = {"source": {"scan": "failed"}}
    assert fh.caption_placement_state(proj) == "failed"
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    proj.caption_placement = {"source": {"scan": "pending", "derived_at": old}}
    assert fh.caption_placement_state(proj, datetime.now(timezone.utc)) == "stale"
    assert fh.caption_placement_state(proj) == "pending"


def test_create_caption_reserves_under_its_own_id(fenv):
    env = fenv
    db = env.SF()
    p = VideoProject(source_type="upload", source_ref="x", status="ready", project_type="localisation")
    db.add(p)
    db.flush()
    final = media_file(env, 2.0, FEMALE)
    run = Run(project_id=p.id, prompt="p", status=RunStatus.done, result_local_path=final)
    db.add(run)
    db.commit()
    resp = api_v2_module.create_caption(run.id, language=None, source_tail_edit_id=None, db=db,
                                        **fh.caption_placement_kwargs())
    scopes = [r.scope_key for r in db.execute(select(KieSpendReservation).where(
        KieSpendReservation.run_id == run.id)).scalars()]
    assert scopes == [f"caption:{resp.caption_id}"]
    db.close()
