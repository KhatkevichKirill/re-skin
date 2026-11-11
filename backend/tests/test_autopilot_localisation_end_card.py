"""AICP-390: the source's own CTA end card in the localisation autopilot.

Pure pieces on project ecb9ac03's real inputs (its transcribe/v3 transcript,
its segments after the AICP-347 re-cut, its scene cuts): the end card is
proposed at the scene cut 49.067 s and the cut is one segment edit. The flow on
the real core (C1's harness, project 46cc6443 + an appended end card): flow v2
asks, the approved cut makes the card a keep segment, the language with a CTA
tail leaves it out of its run, the language without one keeps it as footage;
"generate" keeps today's behaviour; an edit while the question is open asks
again; flow v1 never asks.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.autopilot import core
from app.autopilot.flows import localisation as flow_mod
from app.autopilot.flows import localisation_plan as lp
from app.models import LibraryClip, SegmentDef, VideoProject
from app.models_autopilot import AutopilotApproval
from app.state_machine import ProjectStatus

from tests.test_autopilot_localisation_flow import (  # noqa: F401 — fixtures
    DURATION, SF, answer, engine, env, finish_run, pending, q, runs, settle, steps,
)
from tests.test_autopilot_localisation_plan import SEGMENTS, transcript_v3

EC = json.loads((Path(__file__).parent / "fixtures" / "localisation_ecb9ac03_end_card.json")
                .read_text("utf-8"))
CUT = 49.06666666666667


def ec_segments(ranges=None):
    return [{"id": f"s{i}", "index": i, "start_sec": a, "end_sec": b, "action": "swap",
             "pre_roll_sec": 0.0, "post_roll_sec": 0.0}
            for i, (a, b) in enumerate(ranges or EC["segments"])]


def detect(transcript=None, cuts="default", duration=None):
    return lp.detect_end_card(transcript or EC["transcript"],
                              scene_cuts=EC["scene_cuts"] if cuts == "default" else cuts,
                              duration_sec=duration or EC["duration_sec"])


def plan(segs, card, transcript=None):
    return lp.end_card_plan(segs, card, lines=(transcript or EC["transcript"])["lines"],
                            duration_sec=EC["duration_sec"], target_sec=14.72, floor_sec=4.0)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetect:
    def test_ecb9ac03_end_card_starts_at_the_scene_cut(self):
        card = detect()
        assert card["start_sec"] == pytest.approx(CUT)
        assert card["scene_cut"] is True and card["notes"] == []
        assert card["speakers"] == ["off-screen narrator"]
        assert [ln["text"] for ln in card["lines"]] == ["Listen to anything. Download Listenify now."]
        assert "Listen to anything" in card["on_screen_text"][0]

    def test_without_scene_cuts_it_starts_at_its_first_line_and_says_so(self):
        card = detect(cuts=None)
        assert card["start_sec"] == pytest.approx(49.0) and card["scene_cut"] is False
        assert any("check where the card really starts" in n for n in card["notes"])

    def test_a_closing_line_of_a_voice_heard_before_is_not_a_card(self):
        # 46cc6443 ends with the off-screen interviewer's "Thank you."
        assert lp.detect_end_card(transcript_v3(), scene_cuts=[], duration_sec=DURATION) is None

    def test_a_device_voice_is_not_a_card(self):
        tr = copy.deepcopy(EC["transcript"])
        for spk in tr["speakers"]:
            if spk["tag"] == "NARRATOR":
                spk["kind"] = "device"
        assert detect(tr) is None

    def test_an_on_screen_line_is_not_a_card(self):
        tr = copy.deepcopy(EC["transcript"])
        tr["lines"][-1]["on_screen"] = True
        assert detect(tr) is None

    def test_missing_face_visible_still_counts_as_off_screen(self):
        tr = copy.deepcopy(EC["transcript"])
        tr["lines"][-1].pop("face_visible", None)
        assert detect(tr)["start_sec"] == pytest.approx(CUT)

    def test_a_long_off_screen_ending_is_content_not_a_card(self):
        assert detect(duration=CUT + lp.END_CARD_MAX_SEC + 1.0) is None

    def test_a_whole_video_of_one_off_screen_voice_is_not_a_card(self):
        tr = copy.deepcopy(EC["transcript"])
        tr["lines"] = tr["lines"][-1:]
        assert detect(tr) is None


# ---------------------------------------------------------------------------
# The segment edit
# ---------------------------------------------------------------------------


class TestPlan:
    def test_aligned_boundary_only_turns_the_last_segment_keep(self):
        p = plan(ec_segments(), detect())
        assert p["feasible"] and p["problems"] == []
        assert p["patch"] == {"updates": [{"id": "s4", "start_sec": CUT, "end_sec": EC["duration_sec"],
                                           "action": "keep", "pre_roll_sec": 0.0,
                                           "post_roll_sec": 0.0}],
                              "deletes": [], "creates": []}
        assert [r[2] for r in p["result"]] == ["swap"] * 4 + ["keep"]

    def test_card_inside_a_segment_splits_it_and_a_short_swap_piece_joins_the_previous(self):
        segs = ec_segments([[0.0, 12.9], [12.9, 21.0], [21.0, 34.5], [34.5, 47.0], [47.0, 54.466667]])
        p = plan(segs, detect())
        assert p["feasible"], p["problems"]
        # [47.0, 49.07] is 2.07 s, under the 4 s floor: s3 runs to the card instead.
        assert p["patch"]["updates"] == [{"id": "s3", "start_sec": 34.5, "end_sec": CUT, "action": "swap",
                                          "pre_roll_sec": 0.0, "post_roll_sec": 0.0}]
        assert p["patch"]["deletes"] == ["s4"]
        assert p["patch"]["creates"] == [{"start_sec": CUT, "end_sec": EC["duration_sec"], "action": "keep",
                                          "pre_roll_sec": 0.0, "post_roll_sec": 0.0}]
        assert p["result"][-2:] == [[34.5, round(CUT, 3), "swap"], [round(CUT, 3), 54.467, "keep"]]

    def test_a_long_enough_piece_stays_its_own_swap(self):
        segs = ec_segments([[0.0, 12.9], [12.9, 21.0], [21.0, 34.5], [34.5, 42.0], [42.0, 54.466667]])
        p = plan(segs, detect())
        assert p["feasible"]
        assert p["patch"]["updates"][0]["id"] == "s4" and p["patch"]["updates"][0]["end_sec"] == CUT
        assert p["patch"]["deletes"] == [] and len(p["patch"]["creates"]) == 1

    def test_the_swap_before_the_card_loses_its_post_roll(self):
        segs = ec_segments()
        segs[3]["post_roll_sec"] = 1.0
        p = plan(segs, detect())
        upd = {u["id"]: u for u in p["patch"]["updates"]}
        assert upd["s3"]["post_roll_sec"] == 0.0

    def test_a_line_running_past_the_cut_makes_it_infeasible(self):
        tr = copy.deepcopy(EC["transcript"])
        thanks = next(ln for ln in tr["lines"] if ln["text"].startswith("Thank you"))
        thanks["end"] = 49.8
        card = detect(tr)
        p = plan(ec_segments(), card, tr)
        assert not p["feasible"]
        assert any("runs past the cut" in x for x in p["problems"])

    def test_a_merge_over_the_target_is_infeasible(self):
        segs = ec_segments([[0.0, 12.9], [12.9, 21.0], [21.0, 33.0], [33.0, 47.5], [47.5, 54.466667]])
        p = plan(segs, detect())  # [47.5, 49.07] joins [33, 47.5] -> 16.07 s > 14.72 s
        assert not p["feasible"]
        assert any("over the" in x for x in p["problems"])

    def test_post_roll_is_dropped_before_the_floor_check(self):
        # [46.07, 49.07] is 3 s + 1 s post-roll: 4 s as planned, 3 s once the
        # post-roll stops at the card -> under the floor, it joins the swap before.
        segs = ec_segments([[0.0, 12.9], [12.9, 21.0], [21.0, 34.5], [34.5, CUT - 3.0], [CUT - 3.0, 54.466667]])
        segs[4]["post_roll_sec"] = 1.0
        p = plan(segs, detect())
        assert p["feasible"], p["problems"]
        assert p["patch"]["deletes"] == ["s4"]
        assert p["result"][-2] == [34.5, round(CUT, 3), "swap"]

    def test_pre_roll_before_the_video_start_does_not_count_toward_the_floor(self):
        # A 3 s first swap + 1 s pre-roll is 4 s on paper, 3 s once clamped at 0.
        tr = {"speakers": [{"label": "A", "kind": "person"}, {"label": "N", "kind": "person"}],
              "lines": [{"id": 1, "start": 0.5, "end": 2.5, "speaker": "A", "on_screen": True, "text": "hi"},
                        {"id": 2, "start": 3.2, "end": 6.0, "speaker": "N", "on_screen": False,
                         "face_visible": False, "text": "Download it now."}]}
        card = lp.detect_end_card(tr, scene_cuts=[3.0], duration_sec=7.0)
        segs = [{"id": "a", "index": 0, "start_sec": 0.0, "end_sec": 3.0, "action": "swap",
                 "pre_roll_sec": 1.0, "post_roll_sec": 0.0},
                {"id": "b", "index": 1, "start_sec": 3.0, "end_sec": 7.0, "action": "swap",
                 "pre_roll_sec": 0.0, "post_roll_sec": 0.0}]
        p = lp.end_card_plan(segs, card, lines=tr["lines"], duration_sec=7.0, target_sec=14.72,
                             floor_sec=4.0)
        assert not p["feasible"]
        assert any("under the 4.00 s" in x for x in p["problems"])

    def test_a_short_card_line_with_its_midpoint_before_the_cut_is_infeasible(self):
        tr = copy.deepcopy(EC["transcript"])
        tr["lines"][-1]["start"], tr["lines"][-1]["end"] = 48.9, 49.2  # midpoint 49.05 < 49.067
        card = detect(tr)
        p = plan(ec_segments(), card, tr)
        assert not p["feasible"]
        assert any("before the cut" in x for x in p["problems"])

    def test_segment_ids_read_the_cut_back(self):
        segs = ec_segments()
        assert lp.end_card_segment_ids(segs, CUT) is None  # still swap
        segs[4]["action"] = "keep"
        assert lp.end_card_segment_ids(segs, CUT) == ["s4"]
        segs[3]["post_roll_sec"] = 0.5  # a swap clip running into the card
        assert lp.end_card_segment_ids(segs, CUT) is None


def test_rest_run_spec_leaves_the_excluded_card_out():
    segs = ec_segments()
    segs[4]["action"] = "keep"
    prompts = {"probe_segment_id": "s0", "run_prompt": "p", "segment_prompts": {"s1": "a"}}
    common = dict(project_id="ecb9ac03", language="es", prompts=prompts, segments=segs,
                  probe_run_id="r", temp_clip_id="c")
    with_tail = lp.rest_run_spec(**common, exclude_segment_ids=["s4"])
    assert [e.get("segment_def_id") for e in with_tail["stitch_plan"]] == [None, "s1", "s2", "s3"]
    without = lp.rest_run_spec(**common)
    assert [e.get("segment_def_id") for e in without["stitch_plan"]] == [None, "s1", "s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# The flow on the real core (C1's harness: 46cc6443 + an appended end card)
# ---------------------------------------------------------------------------

CARD_SEC = 5.4
CARD_DURATION = DURATION + CARD_SEC


def card_transcript() -> dict:
    tr = transcript_v3()
    tr["speakers"].append({"label": "off-screen narrator", "tag": "NARRATOR", "kind": "person",
                           "gender": "male", "voice": "deep", "description": "voiceover on the end card"})
    tr["lines"].append({"id": 999, "start": round(DURATION + 0.2, 3), "end": round(DURATION + 4.0, 3),
                        "speaker": "off-screen narrator", "on_screen": False, "face_visible": False,
                        "text": "Listen to anything. Download Listenify now.",
                        "frame": "a promotional graphic of the app"})
    return tr


@pytest.fixture()
def cenv(env, monkeypatch):
    monkeypatch.setattr(flow_mod, "_scene_cuts", lambda ctx: [DURATION])
    monkeypatch.setattr(flow_mod, "_clip_lookup", lambda cid: _lookup(env, cid))
    from app.autopilot import registry

    registry.register(flow_mod.FLOW_V1)
    return env


def _lookup(env, cid):
    db = env.SF()
    try:
        c = db.get(LibraryClip, cid)
        return None if c is None else {"id": c.id, "language": c.language}
    finally:
        db.close()


def make_card_project(env, *, languages=("es",), tails=None, flow=None, transcript=None):
    db = env.SF()
    src = env.tmp / f"src-{os.urandom(4).hex()}.mp4"
    src.write_bytes(os.urandom(256))
    p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                     status=ProjectStatus.ready, project_type="localisation", mode="assistant",
                     localisation_scope="whole", duration_sec=CARD_DURATION, mute_source=True,
                     transcript=transcript or card_transcript(), transcript_status="ready")
    db.add(p)
    db.flush()
    for s in SEGMENTS:
        db.add(SegmentDef(project_id=p.id, index=s["index"], start_sec=s["start_sec"],
                          end_sec=s["end_sec"], action="swap", pre_roll_sec=0.0, post_roll_sec=0.0))
    db.add(SegmentDef(project_id=p.id, index=len(SEGMENTS), start_sec=DURATION,
                      end_sec=CARD_DURATION, action="swap", pre_roll_sec=0.0, post_roll_sec=0.0))
    cta = LibraryClip(id=str(uuid.uuid4()), name="CTA_Spanish", local_path=str(src), duration_sec=7.0,
                      width=96, height=160, fps=12.0, language="es")
    db.add(cta)
    db.commit()
    flow = flow or flow_mod.FLOW
    params = {"languages": list(languages)}
    params["tails"] = {"es": cta.id} if tails is None else tails(cta.id)
    params = flow.validate_params(p, params)
    ap = core.create_autopilot(db, p, flow, params)
    db.commit()
    from types import SimpleNamespace

    out = SimpleNamespace(pid=p.id, apid=ap.id, cta=cta.id)
    db.close()
    return out


def segs_of(env, prj):
    return q(env, SegmentDef, project_id=prj.pid)


def ordered(env, prj):
    return sorted(segs_of(env, prj), key=lambda s: s.index)


def go(env, prj, until, rounds=40):
    """settle(), answering every budget approval on the way."""
    for _ in range(rounds):
        settle(env, prj, rounds=1)
        for b in pending(env, prj, "budget"):
            answer(env, prj, b)
        if until():
            return
    raise AssertionError("go: condition never met")


def drive_to_end_card(env, prj):
    budget = None
    for _ in range(40):
        settle(env, prj, rounds=1)
        b = pending(env, prj, "budget")
        if b:
            budget = b[0]
            answer(env, prj, budget)
        h = [a for a in pending(env, prj, "handoff") if a.name.startswith("endcard:")]
        if h:
            return h[0]
    raise AssertionError("no end-card question")


def test_v2_cut_leaves_the_card_out_of_the_tail_language_only(cenv):
    env = cenv
    prj = make_card_project(env, languages=("es", "pt"), tails=lambda cid: {"es": cid, "pt": None})
    ask = drive_to_end_card(env, prj)
    assert ask.proposal["start_sec"] == pytest.approx(DURATION)
    assert ask.proposal["per_language"]["es"].startswith("cut; CTA_Spanish")
    assert "original footage" in ask.proposal["per_language"]["pt"]
    assert [o["id"] for o in ask.options] == ["replace", "generate", "stop"]
    # Nothing paid before the answer: no run.
    assert runs(env, prj) == []

    answer(env, prj, ask, option="replace")
    settle(env, prj, until=lambda: steps(env, prj).get("segments") is not None
           and steps(env, prj)["segments"].status == "completed")
    st = steps(env, prj)["segments"]
    card = ordered(env, prj)[-1]
    assert card.action == "keep" and card.start_sec == pytest.approx(DURATION)
    assert st.output["end_card"]["choice"] == "replace"
    assert st.output["end_card"]["segment_ids"] == [card.id]

    # es: probe, then the rest run without the card. (A ready transcript puts
    # the budget approval at translate, after the card question.)
    go(env, prj, until=lambda: len(runs(env, prj)) == 1)
    probe_es = runs(env, prj)[0]
    finish_run(env, probe_es.id)
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_probe")))
    answer(env, prj, pending(env, prj, "listening_probe")[0])
    go(env, prj, until=lambda: len(runs(env, prj)) >= 3)  # rest es + probe pt
    by_name = {r.name: r for r in runs(env, prj)}
    rest_es = by_name["Autopilot es"]
    seg_ids = [e.get("segment_def_id") for e in rest_es.stitch_plan]
    assert card.id not in seg_ids
    assert seg_ids[1:] == [s.id for s in ordered(env, prj)[1:-1]]

    # pt (no tail): the card stays in as source footage.
    probe_pt = by_name["Autopilot pt probe"]
    finish_run(env, probe_pt.id)
    go(env, prj, until=lambda: bool(pending(env, prj, "listening_probe")))
    answer(env, prj, pending(env, prj, "listening_probe")[0])
    go(env, prj, until=lambda: "Autopilot pt" in {r.name for r in runs(env, prj)})
    rest_pt = next(r for r in runs(env, prj) if r.name == "Autopilot pt")
    assert rest_pt.stitch_plan[-1] == {"kind": "segment", "segment_def_id": card.id}
    # The card is never generated: no prompt for it in either run.
    db = env.SF()
    try:
        from app.models import RunSegment

        assert db.execute(select(RunSegment).where(RunSegment.segment_def_id == card.id,
                                                   RunSegment.prompt_override.is_not(None))).first() is None
    finally:
        db.close()


def test_v2_generate_keeps_todays_behaviour(cenv):
    env = cenv
    prj = make_card_project(env)
    ask = drive_to_end_card(env, prj)
    answer(env, prj, ask, option="generate")
    settle(env, prj, until=lambda: steps(env, prj)["segments"].status == "completed")
    out = steps(env, prj)["segments"].output
    assert out["end_card"]["choice"] == "generate"
    assert all(s.action == "swap" for s in segs_of(env, prj))


def test_v2_reject_fails_the_attempt(cenv):
    env = cenv
    prj = make_card_project(env)
    ask = drive_to_end_card(env, prj)
    answer(env, prj, ask, "reject", comment="wrong video")
    go(env, prj, until=lambda: steps(env, prj)["segments"].status == "failed")
    assert all(s.action == "swap" for s in segs_of(env, prj))
    assert runs(env, prj) == []


def test_v2_invalid_edit_after_the_answer_fails_with_its_reason(cenv):
    env = cenv
    prj = make_card_project(env)
    ask = drive_to_end_card(env, prj)
    answer(env, prj, ask, option="replace")
    db = env.SF()
    try:  # before the next tick: the first segment becomes keep (nothing to probe)
        first = db.execute(select(SegmentDef).where(SegmentDef.project_id == prj.pid,
                                                    SegmentDef.index == 0)).scalar_one()
        first.action = "keep"
        db.commit()
    finally:
        db.close()
    go(env, prj, until=lambda: steps(env, prj)["segments"].status == "failed")
    assert "not a usable whole-video cut" in steps(env, prj)["segments"].reason


def test_v2_reopen_while_the_question_is_open_never_recuts(cenv):
    env = cenv
    prj = make_card_project(env)
    first = drive_to_end_card(env, prj)
    db = env.SF()
    try:  # a transcript change reopens transcribe -> segments before it completed
        p = db.get(VideoProject, prj.pid)
        tr = copy.deepcopy(p.transcript)
        tr["lines"][0]["text"] = tr["lines"][0]["text"] + " "
        p.transcript = tr
        db.commit()
    finally:
        db.close()
    second = drive_to_end_card(env, prj)
    assert second.id != first.id
    from app.models_autopilot import AutopilotEffect

    recuts = [e for e in q(env, AutopilotEffect, autopilot_id=prj.apid) if e.name == "recut"]
    assert len(recuts) == 1


def test_v2_edit_while_the_question_is_open_asks_again(cenv):
    env = cenv
    prj = make_card_project(env)
    first = drive_to_end_card(env, prj)
    # The operator moves the card boundary 0.1 s later in the Segment Editor.
    db = env.SF()
    try:
        last = db.execute(select(SegmentDef).where(SegmentDef.project_id == prj.pid)
                          .order_by(SegmentDef.index.desc())).scalars().first()
        prev = db.execute(select(SegmentDef).where(SegmentDef.project_id == prj.pid,
                                                   SegmentDef.index == last.index - 1)).scalar_one()
        prev.end_sec = last.start_sec = DURATION + 0.1
        db.commit()
    finally:
        db.close()
    second = drive_to_end_card(env, prj)
    assert second.name != first.name
    assert q(env, AutopilotApproval, id=first.id)[0].status == "superseded"
    # The detection still proposes the scene cut: the plan moves the boundary back.
    assert second.proposal["start_sec"] == pytest.approx(DURATION)


def test_v2_infeasible_boundary_offers_only_generate(cenv):
    env = cenv
    tr = card_transcript()
    thanks = next(ln for ln in tr["lines"] if ln["text"].strip().startswith("Thank you"))
    thanks["end"] = DURATION + 0.6  # runs past the card's start
    prj = make_card_project(env, transcript=tr)
    ask = drive_to_end_card(env, prj)
    assert [o["id"] for o in ask.options] == ["generate", "stop"]
    assert ask.proposal["problems"]


def test_v1_attempt_never_asks(cenv):
    env = cenv
    prj = make_card_project(env, flow=flow_mod.FLOW_V1)
    for _ in range(40):
        settle(env, prj, rounds=1)
        for b in pending(env, prj, "budget"):
            answer(env, prj, b)
        st = steps(env, prj).get("segments")
        if st is not None and st.status == "completed":
            break
    assert st.status == "completed" and "end_card" not in st.output
    assert pending(env, prj, "handoff") == []
    assert all(s.action == "swap" for s in segs_of(env, prj))


def test_v2_project_without_a_card_completes_without_asking(env):
    from tests.test_autopilot_localisation_flow import make_project

    prj = make_project(env, transcript=transcript_v3())
    for _ in range(40):
        settle(env, prj, rounds=1)
        for b in pending(env, prj, "budget"):
            answer(env, prj, b)
        st = steps(env, prj).get("segments")
        if st is not None and st.status == "completed":
            break
    assert st.output["end_card"] is None
    assert pending(env, prj, "handoff") == []
