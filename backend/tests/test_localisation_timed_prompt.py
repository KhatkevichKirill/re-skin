"""
AICP-335 — timed localisation prompt: speaker gender, per-line timecodes,
off-face cues, silent tail, brand pronunciation and action cues.

The acceptance test replays project 46cc6443 (stored transcript + segments,
read-only copy of prod) against the segment prompts that worked on Seedance
2.5 (tests/fixtures/localisation_46cc6443.json, see its "_about").
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from app import localisation

DATA = json.loads(
    (Path(__file__).parent / "fixtures" / "localisation_46cc6443.json").read_text("utf-8")
)
SEGMENTS = sorted(DATA["segments"], key=lambda s: s["index"])
SEG_GLASSES = "a95328cb-66f3-4317-acd8-a11431c3ec4c"   # seg 5, glasses off at ~4–5 s
SEG_PHONE = "8c6816b7-864a-4467-9030-1a94d375394a"     # seg 2, line 9 over the phone


def transcript_v3() -> dict:
    """The stored v2 transcript plus the v3 fields a transcribe/v3 call adds."""
    tr = copy.deepcopy(DATA["transcript_v2"])
    ann = DATA["v3_annotations"]
    for key, value in ann.items():
        if key != "lines":
            tr[key] = copy.deepcopy(value)
    for line in tr["lines"]:
        line.update(ann["lines"].get(str(line["id"]), {}))
        # Every v3 line states its face visibility; v2 only had on_screen.
        line.setdefault("face_visible", bool(line["on_screen"]))
    return tr


def build_segment_prompts(lang: str, transcript: dict, *, reference_audio=True) -> dict:
    """What the localisation-prompt endpoint builds, segment by segment."""
    lines = DATA["translated"][lang]
    out = {}
    for seg in SEGMENTS:
        mine = [
            ln for ln in lines
            if seg["start_sec"] <= (ln["start"] + ln["end"]) / 2 < seg["end_sec"]
        ]
        out[seg["id"]] = localisation.build_prompt(
            lines=mine,
            source_language="en",
            target_language=lang,
            swap_character=False,
            window=localisation.clip_window(
                seg["start_sec"], seg["end_sec"], seg["pre_roll_sec"],
                seg["post_roll_sec"], DATA["duration_sec"],
            ),
            transcript=transcript,
            all_lines=lines,
            reference_audio=reference_audio,
        )
    return out


# The deliberate differences from the hand-written prompts, and nothing else:
# 1. the primary speaker gets the same explicit gender clause as everyone else
#    (the hand-written WOMAN line had none — only the INTERVIEWER, who was the
#    one turning male, did);
# 2. the off-face cue names the speaker by tag ("the woman"), not by a
#    hand-picked phrase ("the older woman");
# 3. seg 5 carries the glasses action cue that fixed fr/pl in a 608-credit
#    re-take after these prompts were written.
_REF_WOMAN = (
    "- WOMAN: the older woman with glasses. She owns the phone. Mature, lower "
    "voice — the older woman's voice from the reference audio."
)
_NEW_WOMAN = (
    "- WOMAN: the older woman with glasses. She owns the phone. Female, mature, "
    "lower voice. Never a male voice."
)
_ACTION = (
    "Action — reproduce it exactly as in the reference video: at ~4–5 s the "
    "WOMAN takes her glasses OFF. Do not put them back on."
)


def background_line(lang: str) -> str:
    """Line 10 — the app reading a book UNDER the woman — kept and marked
    (operator decision 2026-10-29; the hand-written prompts had deleted it)."""
    text = next(ln["text"] for ln in DATA["translated"][lang] if ln["id"] == 10)
    return (
        "[8.4–13.9] APP (BACKGROUND — it plays quietly at the same time as the "
        "WOMAN's lines, underneath them; nobody stops talking for it and it is "
        f"not lip-synced): {text}"
    )


def expected_with_deltas(lang: str) -> dict:
    out = {}
    for seg_id, prompt in DATA["expected"][lang].items():
        if seg_id == SEG_PHONE:
            head, rest = prompt.split("\n\nEvery word the WOMAN", 1)
            prompt = head + "\n" + background_line(lang) + "\n\nEvery word the WOMAN" + rest
        prompt = prompt.replace(_REF_WOMAN, _NEW_WOMAN)
        prompt = prompt.replace(
            "it is still HER voice, the older woman, NOT the interviewer",
            "it is still HER voice, the woman, NOT the interviewer",
        )
        if seg_id == SEG_GLASSES:
            prompt += "\n\n" + _ACTION
        out[seg_id] = prompt
    return out


class TestAcceptance46cc6443:
    @pytest.mark.parametrize("lang", ["es", "pt", "pl", "fr"])
    def test_matches_the_prompts_that_worked(self, lang):
        generated = build_segment_prompts(lang, transcript_v3())
        expected = expected_with_deltas(lang)
        assert set(generated) == set(expected)
        for seg_id in expected:
            assert generated[seg_id] == expected[seg_id], seg_id

    @pytest.mark.parametrize("lang", ["pt", "pl", "fr"])
    def test_dialogue_block_is_byte_identical_except_the_cue_name(self, lang):
        """Timecodes, order, tags, silence and pronunciation: no deltas at all."""
        generated = build_segment_prompts(lang, transcript_v3())
        for seg_id, ref in DATA["expected"][lang].items():
            ref_dialogue = ref.split("\n\nDialogue", 1)[1].split("\n\n")[0]
            gen_dialogue = generated[seg_id].split("\n\nDialogue", 1)[1].split("\n\n")[0]
            ref_dialogue = ref_dialogue.replace("the older woman,", "the woman,")
            if seg_id == SEG_PHONE:
                ref_dialogue += "\n" + background_line(lang)
            assert gen_dialogue == ref_dialogue

    def test_v2_transcript_still_gets_timecodes_and_order(self):
        """No v3 fields: gender/off-face text missing, timing identical."""
        generated = build_segment_prompts("pt", DATA["transcript_v2"])
        stamp = re.compile(r"^\[(\d+\.\d)–(\d+\.\d)\] ", re.M)
        for seg_id, ref in DATA["expected"]["pt"].items():
            expected = stamp.findall(ref)
            if seg_id == SEG_PHONE:
                expected.append(("8.4", "13.9"))  # line 10, kept as background
            assert stamp.findall(generated[seg_id]) == expected, seg_id
            assert "Never a male voice" in generated[seg_id]  # "woman" in the label
        # The overlapping app read-out (line 10) is recognised from the label
        # and kept, marked background.
        assert "Mutações" in generated["77797888-14c2-457e-9f8a-e9567a41978c"]
        assert "] APP (BACKGROUND — it plays quietly" in generated[SEG_PHONE]
        # No face_visible: line 9 is on_screen=true, so no off-face cue.
        assert "face is" not in generated[SEG_PHONE]

    def test_without_reference_audio_the_header_does_not_claim_one(self):
        generated = build_segment_prompts("pt", transcript_v3(), reference_audio=False)
        for prompt in generated.values():
            assert "\n\nSpeakers:\n- WOMAN:" in prompt
            assert "reference audio" not in prompt

    def test_run_marker_survives(self):
        from app import project_types

        for prompt in build_segment_prompts("es", transcript_v3()).values():
            assert project_types.localisation_mode_of(prompt) == "keep"


def L(id, start, end, speaker, text, **extra):
    line = dict(id=id, start=start, end=end, speaker=speaker, on_screen=True, text=text)
    line.update(extra)
    return line


class TestTimedDialogueEdges:
    def test_no_window_is_the_untimed_format(self):
        lines = [L(1, 0, 2, "the woman", "Hola"), L(2, 2, 3, "the woman", "amigo")]
        prompt = localisation.build_prompt(
            lines=lines, source_language="en", target_language="es", swap_character=False,
        )
        assert prompt.endswith("\n**the woman:**     Hola amigo")

    def test_empty_clip_says_nobody_speaks(self):
        out = localisation.format_timed_dialogue(
            [], window=(10.0, 14.5), all_lines=[L(1, 0, 2, "the woman", "x")],
        )
        assert out.endswith("[0.0–4.5] Nobody speaks. Mouths stay closed.")

    def test_overlapping_and_unsorted_lines_never_give_reversed_windows(self):
        lines = [
            L(2, 3.0, 6.0, "the man in blue", "B"),
            L(1, 2.0, 5.0, "the woman", "A"),
            L(3, 2.5, 2.5, "the woman", "C"),
        ]
        out = localisation.format_timed_dialogue(lines, window=(0.0, 8.0))
        stamps = re.findall(r"^\[(\d+\.\d)–(\d+\.\d)\] (\w+)", out, re.M)
        assert [s[2] for s in stamps] == ["WOMAN", "WOMAN", "MAN", "Nobody"]
        for t0, t1, _ in stamps:
            assert float(t0) <= float(t1)
        # The tail counts from the LATEST end, not the last line's.
        assert "[6.0–8.0] Nobody speaks." in out

    def test_timecodes_follow_pre_roll(self):
        window = localisation.clip_window(10.0, 15.0, pre_roll_sec=1.0, post_roll_sec=0.5)
        assert window == (9.0, 15.5)
        out = localisation.format_timed_dialogue(
            [L(1, 10.0, 12.0, "the woman", "Hola")], window=window,
        )
        assert "[1.0–3.0] WOMAN: Hola" in out
        assert localisation.clip_window(70.0, 78.0, 0, 5.0, duration_sec=78.85) == (70.0, 78.85)

    def test_person_interjection_is_never_background(self):
        lines = [
            L(1, 0.0, 8.0, "the woman", "long line"),
            L(2, 3.0, 3.5, "off-screen interviewer", "wow", on_screen=False),
        ]
        assert localisation.background_lines(lines) == []

    def test_device_fully_under_others_is_background(self):
        tr = copy.deepcopy(DATA["transcript_v2"])
        assert [ln["id"] for ln in localisation.background_lines(tr["lines"], tr)] == [10]
        # Partly in the clear: kept.
        lines = [
            L(1, 0.0, 4.0, "the woman", "a"),
            L(2, 3.0, 6.0, "female voice from the phone app", "b", on_screen=False),
        ]
        assert localisation.background_lines(lines) == []

    def test_two_people_with_the_same_role_get_distinct_tags(self):
        lines = [L(1, 0, 1, "the woman in red", "a"), L(2, 1, 2, "the woman in blue", "b")]
        out = localisation.format_timed_dialogue(lines, window=(0.0, 2.0))
        assert "] WOMAN: a" in out and "] WOMAN 2: b" in out

    def test_unknown_gender_secondary_speaker_count_clause(self):
        lines = [
            L(1, 0, 5, "the woman", "a"),
            L(2, 5, 6, "off-screen interviewer", "b", on_screen=False),
        ]
        out = localisation.format_timed_dialogue(lines, window=(0.0, 6.0))
        assert "- INTERVIEWER: off-screen interviewer. They say ONLY ONE line in this clip." in out

    def test_brand_matched_through_source_text(self):
        line = L(1, 0, 2, "the woman", "スピーチファイ", source_text="It's Listenify")
        out = localisation.format_timed_dialogue([line], window=(0.0, 2.0))
        assert '"LISTEN-ih-fy"' in out

    def test_transcript_brand_and_action_outside_window(self):
        tr = {
            "brands": [{"name": "Acme Reader", "pronunciation": "AK-mee"}],
            "actions": [
                {"start": 20.0, "end": 21.0, "speaker": "the woman", "action": "stands up", "do_not": ""},
                {"start": 1.0, "end": 1.2, "speaker": "the woman", "action": "puts the cup down", "do_not": "pick it up again"},
            ],
        }
        out = localisation.format_timed_dialogue(
            [L(1, 0, 2, "the woman", "Acme Reader es genial")], window=(0.0, 5.0), transcript=tr,
        )
        assert 'Pronunciation: "Acme Reader" is a brand name' in out and '"AK-mee"' in out
        assert "stands up" not in out
        assert "at ~1 s the WOMAN puts the cup down. Do not pick it up again." in out

    def test_operator_renamed_label_falls_back(self):
        tr = transcript_v3()
        lines = [L(9, 19.3, 26.2, "the lady", "texto")]
        out = localisation.format_timed_dialogue(lines, window=(12.0, 26.0), transcript=tr, all_lines=lines)
        assert "- LADY: the lady. Female voice. Never a male voice." in out
        # The id-9 metadata belongs to another label, so no off-face cue.
        assert "face is" not in out


class TestCleaners:
    def test_clean_speakers(self):
        out = localisation.clean_speakers([
            {"label": " the woman ", "tag": "woman!", "kind": "robot", "gender": "F", "voice": 3},
            {"label": "The Woman", "tag": "dup"},
            "junk",
            {"label": ""},
        ])
        assert out == [{
            "label": "the woman", "tag": "WOMAN", "kind": "person", "gender": "unknown",
            "voice": "", "description": "",
        }]

    def test_clean_actions(self):
        out = localisation.clean_actions([
            {"start": 5, "end": 4, "speaker": "x", "action": "takes glasses off.", "do_not": "Don't put them back on."},
            {"start": "nan", "action": "y"},
            {"start": 1, "action": ""},
        ])
        assert out == [{"start": 5.0, "end": 5.0, "speaker": "x", "action": "takes glasses off", "do_not": "put them back on"}]

    def test_line_visibility(self):
        assert localisation.clean_line_visibility({"face_visible": "no", "frame": "  "}) == {}
        assert localisation.clean_line_visibility({"face_visible": False, "frame": " the  phone "}) == {
            "face_visible": False, "frame": "the phone",
        }


class TestTranscribeV3Envelope:
    def test_v3_fields_are_cleaned_onto_the_envelope(self, tmp_path, monkeypatch):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        monkeypatch.setattr(localisation.media, "make_llm_proxy", lambda src, dst: src)
        class FakeKie:
            def upload_file(self, *a, **k):
                return "https://files/c.mp4"

        kie = FakeKie()
        payload = {
            "source_language": "en", "video_summary": "", "scene_context": "",
            "speakers": [{"label": "the woman", "tag": "WOMAN", "kind": "person",
                          "gender": "female", "voice": "low", "description": "d"}],
            "lines": [{"id": 1, "start": 0, "end": 1, "speaker": "the woman", "on_screen": True,
                       "face_visible": False, "frame": "the phone", "text": "hi"}],
            "actions": [{"start": 0.5, "end": 0.7, "speaker": "the woman",
                         "action": "waves", "do_not": ""}],
            "brands": [{"name": "Listenify", "pronunciation": "LISTEN-ih-fy"}],
            "on_screen_text": [],
        }
        kie.chat_completion = lambda **k: json.dumps(payload)
        out = localisation.transcribe_video(str(clip), kie=kie)
        assert out["schema_version"] == 3
        assert out["speakers"][0]["gender"] == "female"
        assert out["lines"][0]["face_visible"] is False
        assert out["lines"][0]["frame"] == "the phone"
        assert out["actions"][0]["action"] == "waves"
        assert out["brands"] == [{"name": "Listenify", "pronunciation": "LISTEN-ih-fy"}]

    def test_schema_requires_the_v3_fields(self):
        schema = localisation._TRANSCRIPT_JSON_SCHEMA["json_schema"]["schema"]
        for key in ("speakers", "actions", "brands"):
            assert key in schema["required"]
        line = schema["properties"]["lines"]["items"]
        assert {"face_visible", "frame"} <= set(line["required"])
        assert "FACE VISIBILITY" in localisation.TRANSCRIBE_PROMPT
        assert "ACTIONS" in localisation.TRANSCRIBE_PROMPT


class TestGateRound1:
    def test_clip_window_mirrors_the_roll_trim(self):
        # 15 s segment, 2 s pre-roll, 15 s model: the pre-roll is trimmed away.
        assert localisation.clip_window(10.0, 25.0, 2.0, 0.0, 60.0, 15.0) == (10.0, 25.0)
        # Post-roll goes first: 10 s segment, 2 s pre + 4 s post, 15 s ceiling.
        assert localisation.clip_window(10.0, 20.0, 2.0, 4.0, 60.0, 15.0) == (8.0, 23.0)
        assert localisation.clip_window(10.0, 20.0, 2.0, 4.0, 60.0, 30.0) == (8.0, 24.0)

    def test_on_camera_comes_from_the_whole_transcript(self):
        tr = {"lines": [
            L(1, 0, 3, "the woman", "a", on_screen=False, face_visible=False, frame="the phone"),
            L(2, 30, 33, "the woman", "b", face_visible=True),
        ]}
        hook = [dict(tr["lines"][0], text="A")]
        out = localisation.format_timed_dialogue(hook, window=(0.0, 3.0), transcript=tr, all_lines=hook)
        assert "WOMAN (her face is not visible — the phone" in out

    def test_generated_suffix_never_collides_with_a_declared_tag(self):
        tr = {"speakers": [
            {"label": "a", "tag": "WOMAN"}, {"label": "b", "tag": "WOMAN"},
            {"label": "c", "tag": "WOMAN 2"},
        ]}
        lines = [L(1, 0, 1, "a", "x"), L(2, 1, 2, "b", "y"), L(3, 2, 3, "c", "z")]
        out = localisation.format_timed_dialogue(lines, window=(0.0, 3.0), transcript=tr)
        tags = re.findall(r"^\[\S+\] ([A-Z0-9 ]+):", out, re.M)
        assert len(set(tags)) == 3


class TestOperatorFollowUps:
    """2026-10-29 answers: header toggle, background kept, model field."""

    def test_reference_audio_header_counts_people(self):
        assert localisation.reference_audio_header(DATA["transcript_v2"]) == (
            "Speakers (the reference audio contains both human voices):"
        )
        one = {"lines": [L(1, 0, 1, "the woman", "a")]}
        assert "this human voice" in localisation.reference_audio_header(one)
        three = {"lines": [L(1, 0, 1, "the woman", "a"), L(2, 1, 2, "the man", "b"),
                           L(3, 2, 3, "the girl", "c")]}
        assert "all 3 human voices" in localisation.reference_audio_header(three)
        assert "this human voice" in localisation.reference_audio_header(None)

    def test_toggle_rewrite_round_trips(self):
        off = build_segment_prompts("pt", transcript_v3(), reference_audio=False)
        on = build_segment_prompts("pt", transcript_v3(), reference_audio=True)
        header = localisation.reference_audio_header(transcript_v3())
        for seg_id in off:
            assert localisation.set_reference_audio_header(off[seg_id], header) == on[seg_id]
            assert localisation.set_reference_audio_header(on[seg_id], None) == off[seg_id]
        untimed = "Keep the person.\n**the woman:**     Hola"
        assert localisation.set_reference_audio_header(untimed, header) == untimed

    def test_background_line_is_not_clamped_and_does_not_clamp(self):
        lines = [
            L(1, 0.0, 4.0, "the woman", "a"),
            L(2, 1.0, 3.0, "female voice from the phone app", "bg", on_screen=False),
            L(3, 4.0, 6.0, "the woman", "b"),
        ]
        out = localisation.format_timed_dialogue(lines, window=(0.0, 6.0))
        assert "[0.0–4.0] WOMAN: a" in out
        assert "[1.0–3.0] APP (BACKGROUND" in out
        assert "[4.0–6.0] WOMAN: b" in out

    def test_resolve_prompt_model(self):
        from app import ai_models, project_types

        default = project_types.spec_for("localisation").default_model
        assert localisation.resolve_prompt_model("localisation") is ai_models.spec_for(default)
        assert localisation.resolve_prompt_model("localisation", "seedance-2-5") is (
            ai_models.AI_MODELS["seedance-2-5"]
        )
        with pytest.raises(ValueError):
            localisation.resolve_prompt_model("localisation", "no-such-model")
