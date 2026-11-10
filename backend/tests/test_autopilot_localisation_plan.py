"""AICP-365 phase 1: the pure pieces of the localisation autopilot flow.

Acceptance (AICP-365): with the 46cc6443 inputs, the built prompts match the
prompts that ran on Seedance 2.5 (refs/*_run_inputs.json, through the AICP-335
fixture), the probe run contains the probe segment only and no voice, and the
second run's stitch plan is the TEMP library clip of the probe + segments 2..N
with the probe's audio as the voice reference.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import localisation, spend
from app.autopilot.flows import localisation_plan as lp
from tests.test_localisation_timed_prompt import (
    DATA,
    SEGMENTS,
    SEG_GLASSES,
    SEG_PHONE,
    expected_with_deltas,
    transcript_v3,
)

PROBE = json.loads(
    (Path(__file__).parent / "fixtures" / "autopilot_localisation_46cc6443_probe.json")
    .read_text("utf-8")
)
PID = "46cc6443-646b-4cc8-ac8a-70a3d6a47d86"
SEG1 = SEGMENTS[0]["id"]
DURATION = DATA["duration_sec"]
REF_HEADER = "Speakers (the reference audio contains both human voices):"


def swap_segments():
    return [dict(s, action="swap") for s in SEGMENTS]


def prompts_for(lang: str) -> dict:
    tr = transcript_v3()
    return lp.build_language_prompts(
        translated_lines=DATA["translated"][lang],
        transcript=tr,
        segments=swap_segments(),
        probe_segment_id=SEG1,
        source_language="en",
        target_language=lang,
        duration_sec=DURATION,
    )


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class Clip(SimpleNamespace):
    pass


CLIPS = {
    "cta-es": Clip(id="cta-es", language="es"),
    "cta-any": Clip(id="cta-any", language=None),
}


class TestParams:
    def test_normalises(self):
        got = lp.validate_params(
            {"languages": ["ES", "pt"], "tails": {"es": "cta-es", "pt": ""}},
            clip_lookup=CLIPS.get,
        )
        assert got == {
            "languages": ["es", "pt"],
            "tails": {"es": "cta-es", "pt": None},
            "model": "seedance-2-5",
            "resolution": "720p",
        }

    def test_comma_string_and_untagged_tail(self):
        got = lp.validate_params({"languages": "fr,pl", "tails": {"pl": "cta-any"}},
                                 clip_lookup=CLIPS.get)
        assert got["languages"] == ["fr", "pl"]
        assert got["tails"] == {"fr": None, "pl": "cta-any"}

    @pytest.mark.parametrize("raw, needle", [
        (None, "languages is required"),
        ({"languages": []}, "languages is required"),
        ({"languages": ["nl"]}, "unknown language"),
        ({"languages": ["es", "es"]}, "listed twice"),
        ({"languages": ["es"], "budget": 5}, "unknown autopilot_params"),
        ({"languages": ["es"], "tails": {"pt": None}}, "not one of the languages"),
        ({"languages": ["es"], "tails": {"es": "nope"}}, "unknown library clip"),
        ({"languages": ["pt"], "tails": {"pt": "cta-es"}}, "tagged 'es'"),
        ({"languages": ["es"], "tails": {"es": 3}}, "library clip id or null"),
        ({"languages": ["es"], "tails": ["x"]}, "tails must be an object"),
        ({"languages": ["es"], "model": "seedance-fast"}, "no voice reference"),
        ({"languages": ["es"], "model": "gemini-omni"}, "produces no audio"),
        ({"languages": ["es"], "model": "nope"}, "unknown model"),
        ({"languages": ["es"], "resolution": "4k"}, "resolution"),
        ("es", "must be a JSON object"),
    ])
    def test_refusals(self, raw, needle):
        with pytest.raises(lp.FlowParamsError, match=needle):
            lp.validate_params(raw, clip_lookup=CLIPS.get)

    def test_wrong_project_type(self):
        with pytest.raises(lp.FlowParamsError, match="localisation projects"):
            lp.validate_params({"languages": ["es"]}, project_type="subtitle_removal")

    def test_source_language_check(self):
        params = {"languages": ["es", "en"]}
        assert lp.check_languages_against_source(params, "en") == [
            "target language 'en' is the source language of the video"
        ]
        assert lp.check_languages_against_source(params, None) == []


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


def project(**kw):
    base = dict(
        project_type="localisation", localisation_scope="whole", mute_source=True,
        duration_sec=DURATION, transcript_status="ready", transcript=transcript_v3(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestIntake:
    def test_ready_project(self):
        rep = lp.intake(project())
        assert rep.blockers == []
        assert rep.transcript == "ready"
        assert not rep.needs_mute_source
        assert "79 s long" in rep.warnings[0]

    def test_hook_scope_blocks_and_mute_source_flagged(self):
        rep = lp.intake(project(localisation_scope="hook", mute_source=False, duration_sec=30))
        assert any("hook scope" in b for b in rep.blockers)
        assert rep.needs_mute_source
        assert rep.warnings == []

    @pytest.mark.parametrize("status, transcript, state", [
        (None, None, "missing"),
        ("pending", None, "pending"),
        ("running", None, "pending"),
        ("failed", None, "failed"),
        ("empty", {"lines": []}, "empty"),
        ("ready", DATA["transcript_v2"], "stale"),
        ("ready", {"schema_version": 3, "lines": [{}]}, "ready"),
    ])
    def test_transcript_state(self, status, transcript, state):
        assert lp.transcript_state(status, transcript) == state

    def test_empty_transcript_blocks(self):
        rep = lp.intake(project(transcript_status="empty", transcript={"lines": []}))
        assert rep.transcript == "empty"
        assert any("no speech" in b for b in rep.blockers)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


class TestPartition:
    def test_46cc6443_partition_is_valid(self):
        chk = lp.check_partition(swap_segments(), duration_sec=DURATION, target_sec=14.72)
        assert chk.ok, chk.problems

    def test_over_target_gap_and_keep_first(self):
        segs = swap_segments()
        segs[1] = dict(segs[1], end_sec=segs[1]["start_sec"] + 15.0)   # 15 s and a gap
        segs[0] = dict(segs[0], action="keep")
        chk = lp.check_partition(segs, duration_sec=DURATION, target_sec=14.72)
        assert not chk.ok
        assert chk.over_target == [segs[1]["id"]]
        assert chk.keep_segments == [segs[0]["id"]]
        text = " ".join(chk.problems)
        assert "not contiguous" in text and "nothing to probe" in text

    def test_rolls_count_against_the_target(self):
        segs = swap_segments()
        segs[1] = dict(segs[1], pre_roll_sec=1.0)   # 13.9 + 1.0 > 14.72
        chk = lp.check_partition(segs, duration_sec=DURATION, target_sec=14.72)
        assert chk.over_target == [segs[1]["id"]]

    def test_no_segments(self):
        assert not lp.check_partition([], duration_sec=10, target_sec=14.72).ok


class _Plan:
    def __init__(self, ranges):
        self.ranges = ranges

    def to_dict(self):
        return {"feasible": True, "changed": True, "ranges": self.ranges,
                "warnings": ["w"], "boundaries": [{"index": 0}]}


class TestRecutBoundary:
    def test_injected_recut_is_called_with_the_partition(self):
        seen = {}

        def fake(segs, lines, **kw):
            seen.update(kw, n=len(segs), lines=len(lines))
            return _Plan([[0.0, 12.0], [12.0, DURATION]])

        res = lp.plan_recut(SEGMENTS[:2], transcript_v3()["lines"], duration_sec=DURATION,
                            target_sec=14.72, floor_sec=4.0, scene_cuts=[12.0], recut=fake)
        assert res.available and res.changed and res.feasible
        assert res.ranges == [(0.0, 12.0), (12.0, DURATION)]
        assert seen["n"] == 2 and seen["lines"] == 23
        assert seen["target_sec"] == 14.72 and seen["floor_sec"] == 4.0
        assert seen["scene_cuts"] == [12.0]

    def test_stub_when_speech_recut_is_absent(self, monkeypatch):
        monkeypatch.setattr(lp, "_speech_recut", lambda: None)
        res = lp.plan_recut(swap_segments(), [], duration_sec=DURATION, target_sec=14.72)
        assert not res.available and not res.changed and res.feasible
        assert res.ranges == [(s["start_sec"], s["end_sec"]) for s in SEGMENTS]
        assert "AICP-347" in res.warnings[0]

    def test_real_speech_recut_when_merged(self):
        if lp._speech_recut() is None:
            pytest.skip("app.speech_recut (AICP-347) is not on this branch yet")
        res = lp.plan_recut(swap_segments(), transcript_v3()["lines"],
                            duration_sec=DURATION, target_sec=14.72)
        assert res.available and res.feasible
        assert len(res.ranges) == len(SEGMENTS)
        assert all(e - s <= 14.72 + 1e-6 for s, e in res.ranges)


# ---------------------------------------------------------------------------
# Line assignment + breakdown
# ---------------------------------------------------------------------------


class TestAssignLines:
    def test_same_partition_as_the_translate_job(self):
        from app.api_v2 import _assign_lines_to_segments

        lines = transcript_v3()["lines"]
        segs = [SimpleNamespace(**s) for s in swap_segments()]
        ref_by, ref_dropped = _assign_lines_to_segments(lines, segs)
        got_by, got_dropped = lp.assign_lines(lines, swap_segments())
        assert {k: [l["id"] for l in v] for k, v in got_by.items()} == {
            k: [l["id"] for l in v] for k, v in ref_by.items()
        }
        assert got_dropped == ref_dropped == []

    def test_keep_segment_lines_are_dropped(self):
        segs = swap_segments()
        segs[6] = dict(segs[6], action="keep")
        by, dropped = lp.assign_lines(transcript_v3()["lines"], segs)
        assert segs[6]["id"] not in by
        assert {l["id"] for l in dropped} == {20, 21, 22, 23}


class TestBreakdown:
    def setup_method(self):
        self.b = lp.source_breakdown(transcript_v3(), swap_segments(), duration_sec=DURATION)
        self.by = {s["id"]: s for s in self.b["segments"]}

    def test_speakers_and_genders_from_v3(self):
        tags = {s["tag"]: s for s in self.b["speakers"]}
        assert tags["WOMAN"]["gender"] == "female" and tags["WOMAN"]["primary"]
        assert tags["INTERVIEWER"]["gender"] == "female" and not tags["INTERVIEWER"]["primary"]
        assert tags["APP"]["kind"] == "device"
        assert self.b["schema_version"] == 3
        assert self.b["source_language"] == "en"

    def test_off_face_line_and_background_voice(self):
        seg2 = self.by[SEG_PHONE]
        assert seg2["off_face_line_ids"] == [9]
        line9 = next(r for r in seg2["lines"] if r["id"] == 9)
        assert line9["face_visible"] is False and "over her shoulder" in line9["frame"]
        assert seg2["background_line_ids"] == [10]
        assert self.b["background_line_ids"] == [10]
        counts = {p["tag"]: p["lines"] for p in seg2["persons"]}
        assert counts == {"WOMAN": 2, "INTERVIEWER": 1}

    def test_action_window_is_relative_to_the_clip(self):
        seg5 = self.by[SEG_GLASSES]
        assert seg5["actions"] == [{
            "rel_start_sec": 4.15, "rel_end_sec": 4.95,
            "speaker": "older woman with glasses", "action": "takes her glasses OFF",
            "do_not": "put them back on",
        }]
        assert all(not s.get("actions") for sid, s in self.by.items() if sid != SEG_GLASSES)

    def test_brand_only_where_spoken(self):
        with_brand = [s["index"] for s in self.b["segments"] if s["brands"]]
        assert with_brand == [6]

    def test_speech_and_digest(self):
        assert self.by[SEG1]["speech_sec"] > 10
        assert self.b["transcript_digest"] == lp.transcript_digest(transcript_v3())
        changed = transcript_v3()
        changed["lines"][0]["text"] = "something else"
        assert lp.transcript_digest(changed) != self.b["transcript_digest"]

    def test_v2_transcript_warns(self):
        b = lp.source_breakdown(DATA["transcript_v2"], swap_segments(), duration_sec=DURATION)
        assert any("older than v3" in w for w in b["warnings"])
        tags = {s["tag"]: s["gender"] for s in b["speakers"]}
        assert tags["INTERVIEWER"] == "unknown"

    def test_fits_a_step_output(self):
        assert len(json.dumps(self.b, ensure_ascii=False).encode()) < 64 * 1024


class TestProbeSegment:
    def test_first_segment_with_speech(self):
        assert lp.pick_probe_segment(swap_segments(), transcript_v3()["lines"])["id"] == SEG1

    def test_silent_intro_is_skipped(self):
        lines = [l for l in transcript_v3()["lines"] if l["id"] > 5]
        assert lp.pick_probe_segment(swap_segments(), lines)["id"] == SEGMENTS[1]["id"]

    def test_none_without_speech(self):
        assert lp.pick_probe_segment(swap_segments(), []) is None


# ---------------------------------------------------------------------------
# Prompts — acceptance against the prompts that ran (refs/*_run_inputs.json)
# ---------------------------------------------------------------------------


class TestPromptsAcceptance:
    @pytest.mark.parametrize("lang", ["es", "pt", "pl", "fr"])
    def test_segments_after_the_probe_match_the_references(self, lang):
        got = prompts_for(lang)
        expected = expected_with_deltas(lang)
        assert got["order"] == PROBE["order"]
        assert set(got["segment_prompts"]) == set(PROBE["order"][1:])
        for seg_id in PROBE["order"][1:]:
            assert got["segment_prompts"][seg_id] == expected[seg_id], seg_id
            assert REF_HEADER in got["segment_prompts"][seg_id]
        assert got["dropped_line_ids"] == []

    @pytest.mark.parametrize("lang", ["pt", "pl", "fr"])
    def test_probe_matches_the_probe_that_ran(self, lang):
        """The ref probe = the seg-1 prompt without any voice-reference claim;
        the only delta is AICP-335's explicit gender clause for the WOMAN."""
        got = prompts_for(lang)["probe_prompt"]
        ref = PROBE["probe"][lang].replace(
            "- WOMAN: the older woman with glasses. She owns the phone. Mature, lower voice.",
            "- WOMAN: the older woman with glasses. She owns the phone. Female, mature, "
            "lower voice. Never a male voice.",
        )
        assert got == ref
        assert "reference audio" not in got
        assert "\nSpeakers:\n" in got

    def test_probe_is_the_seg1_prompt_with_the_header_off(self):
        got = prompts_for("pt")
        rebuilt = localisation.set_reference_audio_header(
            expected_with_deltas("pt")[SEG1], None
        )
        assert got["probe_prompt"] == rebuilt

    def test_run_prompt_is_timed_from_the_video_start(self):
        run_prompt = prompts_for("pt")["run_prompt"]
        assert REF_HEADER in run_prompt
        assert "[0.0–2.4] WOMAN: Acho que leio bem mais livros por isso." in run_prompt

    def test_probe_must_be_a_swap_segment(self):
        with pytest.raises(ValueError):
            lp.build_language_prompts(
                translated_lines=DATA["translated"]["pt"], transcript=transcript_v3(),
                segments=swap_segments(), probe_segment_id="nope",
                source_language="en", target_language="pt", duration_sec=DURATION,
            )

    def test_operator_cue(self):
        p = prompts_for("pt")["probe_prompt"]
        once = lp.apply_operator_cue(p, "  the interviewer   must be a WOMAN ")
        assert once.endswith(
            "\n\nCorrections from the operator's listening check — follow them exactly:"
            "\n- the interviewer must be a WOMAN"
        )
        twice = lp.apply_operator_cue(once, "say Listenify, not Listenifyed")
        assert twice.count("Corrections from the operator") == 1
        assert twice.endswith("- the interviewer must be a WOMAN\n- say Listenify, not Listenifyed")
        with pytest.raises(ValueError):
            lp.apply_operator_cue(p, "  ")


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------


class TestEstimates:
    def test_46cc6443_language_estimate(self):
        est = lp.estimate_language(swap_segments(), probe_segment_id=SEG1)
        # Seedance 2.5 720p = 80 cr per started second of clip.
        secs = [13, 14, 10, 14, 9, 9, 13]
        assert est["probe"] == 13 * 80
        assert est["rest"] == sum(secs[1:]) * 80
        assert est["translate"] == 3 * spend.llm_credits()
        assert est["total"] == est["probe"] + est["rest"] + est["translate"]

    def test_before_the_transcript_the_probe_is_segment_one(self):
        a = lp.estimate_language(swap_segments(), probe_segment_id=None)
        b = lp.estimate_language(swap_segments(), probe_segment_id=SEG1)
        assert a == b

    def test_keep_segments_cost_nothing(self):
        segs = swap_segments()
        segs[6] = dict(segs[6], action="keep")
        est = lp.estimate_language(segs, probe_segment_id=SEG1)
        assert est["rest"] == (14 + 10 + 14 + 9 + 9) * 80

    def test_regeneration_and_transcription(self):
        assert lp.estimate_probe_regeneration(swap_segments(), probe_segment_id=SEG1) == 13 * 80
        assert lp.estimate_transcription() == spend.llm_credits()


# ---------------------------------------------------------------------------
# Run specs — acceptance: probe = segment 1 only; rest = TEMP clip + 2..N
# ---------------------------------------------------------------------------


class TestRunSpecs:
    def test_probe_run_is_segment_one_only_without_voice(self):
        prompts = prompts_for("pt")
        spec = lp.probe_run_spec(project_id=PID, language="pt", prompts=prompts)
        assert spec["stitch_plan"] == [{"kind": "segment", "segment_def_id": SEG1}]
        assert spec["segment_prompts"] == {SEG1: prompts["probe_prompt"]}
        assert spec["prompt"] == prompts["probe_prompt"]
        assert spec["voice_from_library_clip"] is None
        assert (spec["model"], spec["resolution"], spec["audio_mode"], spec["language"]) == (
            "seedance-2-5", "720p", "seedance", "pt")

    def test_rest_run_reuses_the_probe_and_its_voice(self):
        prompts = prompts_for("pt")
        spec = lp.rest_run_spec(
            project_id=PID, language="pt", prompts=prompts, segments=swap_segments(),
            probe_run_id="probe-run", temp_clip_id="temp-clip",
        )
        assert spec["stitch_plan"] == [{"kind": "library", "clip_id": "temp-clip"}] + [
            {"kind": "segment", "segment_def_id": sid} for sid in PROBE["order"][1:]
        ]
        assert set(spec["segment_prompts"]) == set(PROBE["order"][1:])
        assert SEG1 not in spec["segment_prompts"]
        # Picture and voice from the same immutable copy of the accepted probe.
        assert spec["voice_from_library_clip"] == "temp-clip"
        assert spec["probe_run_id"] == "probe-run"
        assert spec["prompt"] == prompts["run_prompt"]

    def test_probe_in_the_middle_keeps_its_place(self):
        segs = swap_segments()
        lines = [l for l in transcript_v3()["lines"] if l["id"] > 5]
        probe = lp.pick_probe_segment(segs, lines)
        prompts = lp.build_language_prompts(
            translated_lines=[l for l in DATA["translated"]["pt"] if l["id"] > 5],
            transcript=transcript_v3(), segments=segs, probe_segment_id=probe["id"],
            source_language="en", target_language="pt", duration_sec=DURATION,
        )
        spec = lp.rest_run_spec(project_id=PID, language="pt", prompts=prompts,
                                segments=segs, probe_run_id="r", temp_clip_id="c")
        assert spec["stitch_plan"][0] == {"kind": "segment", "segment_def_id": SEG1}
        assert spec["stitch_plan"][1] == {"kind": "library", "clip_id": "c"}
        assert prompts["segment_prompts"][SEG1].count("Nobody speaks") == 1

    def test_temp_clip_is_untagged(self):
        spec = lp.temp_clip_spec(project_id=PID, language="pt", probe_run_id="r")
        assert spec == {"name": "TEMP autopilot 46cc6443 pt probe", "language": None,
                        "from_run": "r", "file": "final"}

    def test_specs_are_json(self):
        prompts = prompts_for("fr")
        for spec in (
            lp.probe_run_spec(project_id=PID, language="fr", prompts=prompts),
            lp.rest_run_spec(project_id=PID, language="fr", prompts=prompts,
                             segments=swap_segments(), probe_run_id="r", temp_clip_id="c"),
        ):
            assert json.loads(json.dumps(spec)) == spec
