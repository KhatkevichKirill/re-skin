"""
Tests for app.localisation — no real network calls, no real API key.

Uses respx to intercept httpx, exactly like test_kie_client.py. The two kie.ai
routes are mocked at their real URLs because the module resolves them from
``KieClient._jobs_base``, and pinning the real paths is half the point: a change
to ``/{model}/v1/chat/completions`` or ``/codex/v1/responses`` should break a
test, not production.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app import kie_client, localisation, project_types
from app.kie_client import KieClient
from app.localisation import (
    LocalisationError,
    build_prompt,
    format_dialogue,
    slice_lines,
    transcribe_video,
    translate_lines,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FAKE_KEY = "test-api-key-not-real"
UPLOAD_BASE = "https://upload.test"
JOBS_BASE = "https://api.kie.ai"

CHAT_URL = f"{JOBS_BASE}/gemini-2.5-pro/v1/chat/completions"
RESPONSES_URL = f"{JOBS_BASE}/codex/v1/responses"
UPLOAD_URL = f"{UPLOAD_BASE}/api/file-stream-upload"


@pytest.fixture
def kie():
    """A KieClient pointed at the test upload host, with a no-op sleep."""
    return KieClient(
        api_key=FAKE_KEY,
        upload_base=UPLOAD_BASE,
        jobs_base=JOBS_BASE,
        sleep_fn=lambda _s: None,
    )


@pytest.fixture
def shared_kie(kie, monkeypatch):
    """Make ``get_shared_client()`` (used by translate_lines) return *kie*."""
    monkeypatch.setattr(kie_client, "_shared_client", kie)
    return kie


@pytest.fixture
def clip(tmp_path):
    src = tmp_path / "hook.mp4"
    src.write_bytes(b"fakevideo")
    return str(src)


def mock_upload() -> None:
    respx.post(UPLOAD_URL).mock(
        return_value=httpx.Response(
            200,
            json={"code": 200, "data": {"downloadUrl": "https://tempfile.test/h.mp4"}},
        )
    )


def chat_response(content: str) -> httpx.Response:
    """The real chat/completions success envelope, observed live."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1786238411,
            "credits_consumed": 0.38,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 2001, "completion_tokens": 373},
        },
    )


def responses_response(text: str, *, with_reasoning: bool = False) -> httpx.Response:
    """The real /codex/v1/responses success envelope, observed live.

    ``with_reasoning`` prepends a reasoning item so the "message is not at a
    fixed index" case is covered — the published example documents
    ``output[1]``, but a live medium-effort call put the message at index 0.
    """
    output = []
    if with_reasoning:
        output.append({"type": "reasoning", "id": "rs_1", "summary": []})
    output.append(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    )
    return httpx.Response(
        200,
        json={"status": "completed", "model": "gpt-5.6-luna", "output": output},
    )


TRANSCRIPT_JSON = {
    "source_language": "EN",
    "video_summary": "  A street-interview promo for a reading app.  ",
    "scene_context": "  Interviewer is off-camera; woman holds a phone labeled Speechify.  ",
    "lines": [
        {
            "id": 1,
            "start": 0.0,
            "end": 1.2,
            "speaker": "off-screen interviewer holding the camera",
            "on_screen": False,
            "text": "What is she doing?",
        },
        {
            "id": 2,
            "start": 5.2,
            "end": 6.8,
            "speaker": "the woman in the pattern shirt",
            "on_screen": True,
            "text": "Oh, I'm not.",
        },
    ],
    "on_screen_text": [{"start": 0.0, "end": 3.0, "text": "READ 3x FASTER"}],
}


def lines(*specs) -> list[dict]:
    """Build line dicts from ``(id, start, end, speaker, text)`` tuples."""
    return [
        {
            "id": i,
            "start": s,
            "end": e,
            "speaker": spk,
            "on_screen": True,
            "text": txt,
        }
        for i, s, e, spk, txt in specs
    ]


# ---------------------------------------------------------------------------
# transcribe_video
# ---------------------------------------------------------------------------


class TestTranscribeVideo:
    @respx.mock
    def test_parses_plain_json(self, kie, clip):
        mock_upload()
        route = respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert route.called
        assert out["schema_version"] == localisation.TRANSCRIPT_SCHEMA_VERSION
        assert out["prompt_version"] == localisation.TRANSCRIBE_PROMPT_VERSION
        assert out["model"] == "gemini-2.5-pro"
        assert out["created_at"].endswith("+00:00")
        # "EN" is normalised to a bare lowercase ISO 639-1 code.
        assert out["source_language"] == "en"
        # Context strings are stripped and stored on the envelope.
        assert out["video_summary"] == "A street-interview promo for a reading app."
        assert out["scene_context"] == (
            "Interviewer is off-camera; woman holds a phone labeled Speechify."
        )
        assert [ln["text"] for ln in out["lines"]] == [
            "What is she doing?",
            "Oh, I'm not.",
        ]
        assert out["on_screen_text"] == [
            {"start": 0.0, "end": 3.0, "text": "READ 3x FASTER"}
        ]

    @respx.mock
    def test_prompt_and_schema_request_video_context(self, kie, clip):
        """Gemini is explicitly asked for video_summary + scene_context."""
        mock_upload()
        route = respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        transcribe_video(clip, kie=kie)

        body = json.loads(route.calls[0].request.content)
        prompt = body["messages"][0]["content"][0]["text"]
        assert "video_summary" in prompt
        assert "scene_context" in prompt
        assert "VIDEO SUMMARY" in prompt
        assert "SCENE CONTEXT" in prompt
        schema = body["response_format"]["json_schema"]["schema"]
        assert "video_summary" in schema["properties"]
        assert "scene_context" in schema["properties"]
        assert "video_summary" in schema["required"]
        assert "scene_context" in schema["required"]

    @respx.mock
    def test_missing_or_null_context_becomes_empty_string(self, kie, clip):
        """A usable lines array must not crash when context fields are absent."""
        mock_upload()
        payload = {
            "source_language": "en",
            # video_summary omitted entirely; scene_context is an unusable type.
            "scene_context": 12,
            "lines": TRANSCRIPT_JSON["lines"],
            "on_screen_text": [],
        }
        respx.post(CHAT_URL).mock(return_value=chat_response(json.dumps(payload)))

        out = transcribe_video(clip, kie=kie)

        assert out["video_summary"] == ""
        assert out["scene_context"] == ""
        assert out["schema_version"] == 2
        assert out["prompt_version"] == "transcribe/v2"

    @respx.mock
    def test_null_context_values_become_empty_string(self, kie, clip):
        mock_upload()
        payload = {
            "source_language": "en",
            "video_summary": None,
            "scene_context": None,
            "lines": TRANSCRIPT_JSON["lines"],
            "on_screen_text": [],
        }
        respx.post(CHAT_URL).mock(return_value=chat_response(json.dumps(payload)))

        out = transcribe_video(clip, kie=kie)

        assert out["video_summary"] == ""
        assert out["scene_context"] == ""

    @respx.mock
    def test_sends_video_in_image_url_envelope(self, kie, clip):
        """kie.ai passes every media type through image_url — including video."""
        mock_upload()
        route = respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        transcribe_video(clip, kie=kie)

        body = json.loads(route.calls[0].request.content)
        assert body["stream"] is False
        assert body["response_format"]["type"] == "json_schema"
        parts = body["messages"][0]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "https://tempfile.test/h.mp4"},
        }

    @respx.mock
    def test_model_override_goes_in_the_url_path(self, kie, clip):
        """The chat route carries the model in the PATH, not the body."""
        mock_upload()
        route = respx.post(f"{JOBS_BASE}/some-other-model/v1/chat/completions").mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie, model="some-other-model")

        assert route.called
        assert "model" not in json.loads(route.calls[0].request.content)
        assert out["model"] == "some-other-model"

    @respx.mock
    def test_tolerates_fenced_json(self, kie, clip):
        """Gemini fences its answer even under a strict response_format schema."""
        mock_upload()
        fenced = "```json\n" + json.dumps(TRANSCRIPT_JSON) + "\n```"
        respx.post(CHAT_URL).mock(return_value=chat_response(fenced))

        out = transcribe_video(clip, kie=kie)

        assert len(out["lines"]) == 2

    @respx.mock
    def test_tolerates_prose_around_the_object(self, kie, clip):
        mock_upload()
        noisy = (
            "Sure! Here is the transcript:\n"
            + json.dumps(TRANSCRIPT_JSON)
            + "\nLet me know if you need anything else."
        )
        respx.post(CHAT_URL).mock(return_value=chat_response(noisy))

        out = transcribe_video(clip, kie=kie)

        assert len(out["lines"]) == 2

    @respx.mock
    def test_empty_transcript_is_a_success(self, kie, clip):
        """A hook with no speech is a legal outcome, not an error (§4.1)."""
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=chat_response(
                json.dumps({"source_language": "", "lines": [], "on_screen_text": []})
            )
        )

        out = transcribe_video(clip, kie=kie)

        assert out["lines"] == []
        assert out["on_screen_text"] == []
        assert out["source_language"] == ""

    @respx.mock
    def test_lines_are_sorted_and_renumbered(self, kie, clip):
        """Ids are the translation rejoin key, so they are rebuilt 1..N in time
        order regardless of what the model numbered."""
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=chat_response(
                json.dumps(
                    {
                        "source_language": "en",
                        "lines": [
                            {"id": 9, "start": 4.0, "end": 5.0,
                             "speaker": "B", "on_screen": True, "text": "second"},
                            {"id": 3, "start": 1.0, "end": 2.0,
                             "speaker": "A", "on_screen": True, "text": "first"},
                            {"id": 7, "start": 6.0, "end": 7.0,
                             "speaker": "C", "on_screen": True, "text": "   "},
                        ],
                        "on_screen_text": [],
                    }
                )
            )
        )

        out = transcribe_video(clip, kie=kie)

        # The blank-text line is dropped; the rest are renumbered in time order.
        assert [(ln["id"], ln["text"]) for ln in out["lines"]] == [
            (1, "first"),
            (2, "second"),
        ]

    @respx.mock
    def test_unparseable_response_raises(self, kie, clip):
        mock_upload()
        respx.post(CHAT_URL).mock(return_value=chat_response("I could not do that."))

        with pytest.raises(LocalisationError, match="could not parse"):
            transcribe_video(clip, kie=kie)

    @respx.mock
    def test_empty_content_raises(self, kie, clip):
        mock_upload()
        respx.post(CHAT_URL).mock(return_value=chat_response(""))

        with pytest.raises(LocalisationError, match="empty response"):
            transcribe_video(clip, kie=kie)

    @respx.mock
    def test_http_200_error_envelope_raises(self, kie, clip):
        """kie.ai reports auth/model errors as HTTP 200 + {"code","msg"}."""
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200, json={"code": 401, "msg": "Unauthorized", "data": None}
            )
        )

        with pytest.raises(LocalisationError, match="code=401"):
            transcribe_video(clip, kie=kie)

    @respx.mock
    def test_http_error_raises(self, kie, clip):
        mock_upload()
        respx.post(CHAT_URL).mock(return_value=httpx.Response(400, text="bad request"))

        with pytest.raises(LocalisationError, match="LLM HTTP 400"):
            transcribe_video(clip, kie=kie)

    @respx.mock
    def test_upload_failure_raises_localisation_error(self, kie, clip):
        respx.post(UPLOAD_URL).mock(return_value=httpx.Response(413, text="too big"))

        with pytest.raises(LocalisationError, match="upload failed"):
            transcribe_video(clip, kie=kie)


# ---------------------------------------------------------------------------
# translate_lines
# ---------------------------------------------------------------------------


SRC = lines(
    (1, 0.0, 1.2, "off-screen interviewer", "What is she doing?"),
    (2, 1.6, 4.8, "the woman in the pattern shirt", "Oh, I'm not."),
)


class TestTranslateLines:
    @respx.mock
    def test_preserves_ids_order_and_source_text(self, shared_kie):
        respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps(
                    {
                        "lines": [
                            {"id": 2, "text": "あ、読んでないんです。"},
                            {"id": 1, "text": "あの子、何してるの？"},
                        ]
                    }
                )
            )
        )

        out = translate_lines(SRC, source_language="en", target_language="ja")

        # Same length, same ids, ORIGINAL order — not the model's order.
        assert [ln["id"] for ln in out] == [1, 2]
        assert out[0]["text"] == "あの子、何してるの？"
        assert out[0]["source_text"] == "What is she doing?"
        assert out[1]["text"] == "あ、読んでないんです。"
        assert out[1]["source_text"] == "Oh, I'm not."
        # Timing and speaker survive untouched.
        assert out[0]["speaker"] == "off-screen interviewer"
        assert (out[1]["start"], out[1]["end"]) == (1.6, 4.8)
        # The input list is not mutated.
        assert SRC[0]["text"] == "What is she doing?"

    @respx.mock
    def test_prompt_carries_durations_and_model_in_body(self, shared_kie):
        """Duration in seconds is the whole point of the translation prompt."""
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps({"lines": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}]})
            )
        )

        translate_lines(SRC, source_language="en", target_language="ja")

        body = json.loads(route.calls[0].request.content)
        # The Responses route carries the model in the BODY and has no
        # response_format at all.
        assert body["model"] == "gpt-5-6-luna"
        assert "response_format" not in body
        assert body["reasoning"] == {"effort": "medium"}
        assert body["stream"] is False
        part = body["input"][0]["content"][0]
        assert part["type"] == "input_text"
        prompt = part["text"]
        assert "duration: 1.2s" in prompt
        assert "duration: 3.2s" in prompt
        assert "from english into japanese" in prompt
        # Omitted context args still render the labeled section (empty → placeholder).
        assert "VIDEO SUMMARY:" in prompt
        assert "SCENE CONTEXT:" in prompt

    @respx.mock
    def test_prompt_includes_video_context_when_supplied(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps({"lines": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}]})
            )
        )

        translate_lines(
            SRC,
            source_language="en",
            target_language="ja",
            video_summary="Promo for a reading app.",
            scene_context="Woman holds a phone; 'this' means the app.",
        )

        prompt = json.loads(route.calls[0].request.content)["input"][0]["content"][0][
            "text"
        ]
        assert "VIDEO SUMMARY:\nPromo for a reading app." in prompt
        assert "SCENE CONTEXT:\nWoman holds a phone; 'this' means the app." in prompt
        # Hard constraints survive the context section.
        assert "ONE LINE IN, ONE LINE OUT" in prompt
        assert "SAY OUT LOUD" in prompt

    @respx.mock
    def test_omitted_context_args_still_translate(self, shared_kie):
        """Backward compatible: callers that never heard of the kwargs keep working."""
        respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps({"lines": [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]})
            )
        )

        out = translate_lines(SRC, source_language="en", target_language="ja")
        assert [ln["text"] for ln in out] == ["a", "b"]

    @respx.mock
    def test_reads_message_after_a_reasoning_item(self, shared_kie):
        respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps({"lines": [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]}),
                with_reasoning=True,
            )
        )

        out = translate_lines(SRC, source_language="en", target_language="ja")

        assert [ln["text"] for ln in out] == ["a", "b"]

    @respx.mock
    def test_tolerates_fenced_json(self, shared_kie):
        """No response_format on this route, so fences are expected."""
        respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                '```json\n{"lines": [{"id": 1, "text": "a"}, '
                '{"id": 2, "text": "b"}]}\n```'
            )
        )

        out = translate_lines(SRC, source_language="en", target_language="ja")

        assert [ln["text"] for ln in out] == ["a", "b"]

    def test_empty_input_short_circuits(self, shared_kie):
        """No lines => no API call at all (respx is not even armed here)."""
        assert translate_lines([], source_language="en", target_language="ja") == []

    @respx.mock
    def test_dropped_line_raises(self, shared_kie):
        """A silently shortened script would desync the hook — fail loudly."""
        respx.post(RESPONSES_URL).mock(
            return_value=responses_response(json.dumps({"lines": [{"id": 1, "text": "a"}]}))
        )

        with pytest.raises(LocalisationError, match="no text for line id=2"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_invented_line_raises(self, shared_kie):
        respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps(
                    {
                        "lines": [
                            {"id": 1, "text": "a"},
                            {"id": 2, "text": "b"},
                            {"id": 3, "text": "bonus"},
                        ]
                    }
                )
            )
        )

        with pytest.raises(LocalisationError, match=r"invented line ids.*\[3\]"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_missing_lines_array_raises(self, shared_kie):
        respx.post(RESPONSES_URL).mock(
            return_value=responses_response(json.dumps({"result": "ok"}))
        )

        with pytest.raises(LocalisationError, match="no 'lines' array"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_http_200_error_envelope_raises(self, shared_kie):
        respx.post(RESPONSES_URL).mock(
            return_value=httpx.Response(200, json={"code": 401, "msg": "Unauthorized"})
        )

        with pytest.raises(LocalisationError, match="code=401"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_no_output_text_raises(self, shared_kie):
        respx.post(RESPONSES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "incomplete",
                    "output": [{"type": "reasoning", "id": "rs_1", "summary": []}],
                },
            )
        )

        with pytest.raises(LocalisationError, match="no output_text"):
            translate_lines(SRC, source_language="en", target_language="ja")


# ---------------------------------------------------------------------------
# translate_lines — strict 1:1 enforcement
# ---------------------------------------------------------------------------


def translated(*pairs) -> str:
    """A translator answer body: ``translated((1, "a"), (2, "b"))``."""
    return json.dumps({"lines": [{"id": i, "text": t} for i, t in pairs]})


class TestTranslateOneToOne:
    """The rejoin is by id, so anything but one entry per requested id fails.

    The dangerous case is a **duplicated** id: the model splits one line into
    two entries that both claim it, and a plain ``by_id[id] = text`` rejoin
    keeps only the last fragment — half a sentence vanishes from the run prompt
    with no error, and the operator finds out after paying for a generation.

    Because a split is a sampling accident on a non-deterministic route rather
    than a bad request, the same prompt is re-asked once before the operator
    sees a 502 (docs/localisation.md §4.4, §8).
    """

    @respx.mock
    def test_duplicate_id_is_retried_then_raises(self, shared_kie):
        """Both fragments of a split line claim id 2 — never last-write-wins."""
        split = json.dumps(
            {
                "lines": [
                    {"id": 1, "text": "あの子、何してるの？"},
                    {"id": 2, "text": "あ、"},          # first half of line 2
                    {"id": 2, "text": "読んでないんです。"},  # second half
                ]
            }
        )
        route = respx.post(RESPONSES_URL).mock(
            side_effect=[responses_response(split), responses_response(split)]
        )

        with pytest.raises(LocalisationError, match=r"duplicate line ids.*id=2 x2"):
            translate_lines(SRC, source_language="en", target_language="ja")

        # Re-asked exactly once, and the message says the re-ask happened so
        # the operator knows "try again" is not the fix.
        assert route.call_count == 2

    @respx.mock
    def test_duplicate_id_error_reports_the_retry(self, shared_kie):
        split = translated((1, "a"), (2, "b"), (2, "b again"))
        respx.post(RESPONSES_URL).mock(
            side_effect=[responses_response(split), responses_response(split)]
        )

        with pytest.raises(LocalisationError, match=r"still wrong after 2 attempts"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_duplicate_id_recovers_on_the_retry(self, shared_kie):
        """One re-ask is why we retry at all: the second draft is usually clean."""
        route = respx.post(RESPONSES_URL).mock(
            side_effect=[
                responses_response(translated((1, "a"), (2, "b"), (2, "b2"))),
                responses_response(translated((1, "clean a"), (2, "clean b"))),
            ]
        )

        out = translate_lines(SRC, source_language="en", target_language="ja")

        assert [ln["id"] for ln in out] == [1, 2]
        assert [ln["text"] for ln in out] == ["clean a", "clean b"]
        assert [ln["source_text"] for ln in out] == [
            "What is she doing?",
            "Oh, I'm not.",
        ]
        assert route.call_count == 2

    @respx.mock
    def test_dropped_line_recovers_on_the_retry(self, shared_kie):
        """The re-ask covers every 1:1 break, not just duplicates."""
        route = respx.post(RESPONSES_URL).mock(
            side_effect=[
                responses_response(translated((1, "a"))),
                responses_response(translated((1, "a"), (2, "b"))),
            ]
        )

        out = translate_lines(SRC, source_language="en", target_language="ja")

        assert [ln["text"] for ln in out] == ["a", "b"]
        assert route.call_count == 2

    @respx.mock
    def test_unparseable_answer_recovers_on_the_retry(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            side_effect=[
                responses_response("sorry, I can't help with that"),
                responses_response(translated((1, "a"), (2, "b"))),
            ]
        )

        out = translate_lines(SRC, source_language="en", target_language="ja")

        assert [ln["text"] for ln in out] == ["a", "b"]
        assert route.call_count == 2

    @respx.mock
    def test_api_failure_is_not_retried(self, shared_kie):
        """The client already retries transient transport errors; 401 will not fix
        itself, and re-asking would only double the operator's wait."""
        route = respx.post(RESPONSES_URL).mock(
            return_value=httpx.Response(200, json={"code": 401, "msg": "Unauthorized"})
        )

        with pytest.raises(LocalisationError, match="code=401"):
            translate_lines(SRC, source_language="en", target_language="ja")

        assert route.call_count == 1

    @respx.mock
    @pytest.mark.parametrize(
        "junk",
        [
            {"id": None, "text": "…and that's the rest of it"},  # no id at all
            {"text": "…and that's the rest of it"},              # id key missing
            {"id": "2b", "text": "…fragment"},                   # unparseable id
            {"id": 2.5, "text": "…fragment"},                    # NOT floor()ed to 2
            {"id": True, "text": "…fragment"},                   # bool is not id 1
            "and that's all folks",                              # not an object
        ],
    )
    def test_entry_with_no_usable_id_raises(self, shared_kie, junk):
        """An entry that claims no line is usually half of a split one.

        That is the text we must not lose, so skipping it (the old behaviour)
        is the same data-loss bug as a duplicated id in a different hat. The
        ``2.5`` case is the nastiest: ``int(2.5)`` used to file the fragment
        against line 2 and overwrite its real translation.
        """
        body = json.dumps(
            {"lines": [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, junk]}
        )
        respx.post(RESPONSES_URL).mock(
            side_effect=[responses_response(body), responses_response(body)]
        )

        with pytest.raises(LocalisationError, match="no usable line id"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_blank_text_for_a_requested_id_still_raises(self, shared_kie):
        """An entry that claims a line but carries no text is a dropped line."""
        body = translated((1, "a"), (2, "   "))
        respx.post(RESPONSES_URL).mock(
            side_effect=[responses_response(body), responses_response(body)]
        )

        with pytest.raises(LocalisationError, match="no text for line id=2"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_duplicate_beats_invented_in_the_message(self, shared_kie):
        """Two ways to be wrong at once: the silent-data-loss one is named first."""
        body = translated((1, "a"), (2, "b"), (2, "b2"), (9, "invented"))
        respx.post(RESPONSES_URL).mock(
            side_effect=[responses_response(body), responses_response(body)]
        )

        with pytest.raises(LocalisationError, match="duplicate line ids"):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_shifted_ids_report_only_the_matched_count(self, shared_kie):
        """The count in the message is matched ids, not entries received.

        With every id shifted by one, "2 of 2 lines came back" would be a lie.
        """
        body = translated((2, "a"), (3, "b"))
        respx.post(RESPONSES_URL).mock(
            side_effect=[responses_response(body), responses_response(body)]
        )

        match = r"id=1.*\(1 of 2 lines came back\)"
        with pytest.raises(LocalisationError, match=match):
            translate_lines(SRC, source_language="en", target_language="ja")

    @respx.mock
    def test_duplicate_ids_in_the_request_raise_before_any_call(self, shared_kie):
        """The request's own ids have to be a usable rejoin key too.

        A transcript hand-edited into two id=2 lines can only come back as one
        entry, and both lines would then be handed the same translation with
        every model-side check passing — so it fails before a call is paid for.
        """
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(translated((1, "a"), (2, "b")))
        )
        broken = lines(
            (1, 0.0, 1.2, "interviewer", "What is she doing?"),
            (2, 1.6, 3.0, "the woman", "Oh, I'm not."),
            (2, 3.0, 4.8, "the woman", "I'm listening."),
        )

        with pytest.raises(LocalisationError, match=r"duplicate ids \[2\]"):
            translate_lines(broken, source_language="en", target_language="ja")

        assert route.call_count == 0

    @respx.mock
    @pytest.mark.parametrize("bad_id", ["1", None, 1.5, True])
    def test_unusable_request_ids_raise_before_any_call(self, shared_kie, bad_id):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(translated((1, "a")))
        )
        broken = [dict(SRC[0], id=bad_id)]

        with pytest.raises(LocalisationError, match="no usable integer id"):
            translate_lines(broken, source_language="en", target_language="ja")

        assert route.call_count == 0


# ---------------------------------------------------------------------------
# slice_lines
# ---------------------------------------------------------------------------


class TestSliceLines:
    WINDOW = lines(
        (1, 0.0, 2.0, "A", "before-and-inside"),
        (2, 4.0, 6.0, "B", "wholly inside"),
        (3, 9.0, 12.0, "C", "straddles the end"),
        (4, 10.0, 11.0, "D", "starts exactly at the end"),
        (5, 12.0, 14.0, "E", "wholly after"),
    )

    def test_returns_overlapping_lines(self):
        got = slice_lines(self.WINDOW, 0.0, 10.0)
        # 3 straddles the boundary and stays: the hook is what gets re-spoken,
        # so half-covered speech must not be silently dropped.
        assert [ln["id"] for ln in got] == [1, 2, 3]

    def test_line_ending_exactly_at_start_is_excluded(self):
        got = slice_lines(lines((1, 0.0, 4.0, "A", "x")), 4.0, 8.0)
        assert got == []

    def test_line_starting_exactly_at_end_is_excluded(self):
        got = slice_lines(lines((1, 8.0, 9.0, "A", "x")), 0.0, 8.0)
        assert got == []

    def test_line_starting_exactly_at_start_is_included(self):
        got = slice_lines(lines((1, 4.0, 5.0, "A", "x")), 4.0, 8.0)
        assert [ln["id"] for ln in got] == [1]

    def test_zero_length_line_inside_window_is_included(self):
        got = slice_lines(lines((1, 3.0, 3.0, "A", "x")), 0.0, 8.0)
        assert [ln["id"] for ln in got] == [1]

    def test_zero_length_line_at_end_boundary_is_excluded(self):
        got = slice_lines(lines((1, 8.0, 8.0, "A", "x")), 0.0, 8.0)
        assert got == []

    def test_empty_and_none_inputs(self):
        assert slice_lines([], 0.0, 10.0) == []
        assert slice_lines(None, 0.0, 10.0) == []

    def test_returns_the_same_dict_objects(self):
        """Slicing is a filter, not a copy — callers rely on identity."""
        got = slice_lines(self.WINDOW, 0.0, 3.0)
        assert got[0] is self.WINDOW[0]


# ---------------------------------------------------------------------------
# format_dialogue
# ---------------------------------------------------------------------------


class TestFormatDialogue:
    def test_matches_the_proven_run_shape(self):
        out = format_dialogue(
            lines(
                (1, 0.0, 1.0, "Camera Person (Interviewer)", "彼女、何してるの？"),
                (2, 2.0, 3.0, "Woman", "あ、読んでないよ。"),
            )
        )
        assert out == (
            "**Camera Person (Interviewer):**     彼女、何してるの？\n"
            "**Woman:**     あ、読んでないよ。"
        )

    def test_merges_consecutive_lines_from_the_same_speaker(self):
        """Run 1736426f merged the interviewer's two opening lines into one
        block; re-labelling each clause makes the model pause between them."""
        out = format_dialogue(
            lines(
                (1, 0.0, 1.0, "Interviewer", "What is she doing?"),
                (2, 1.6, 4.8, "Interviewer", "How are you reading and walking?"),
                (3, 5.0, 6.0, "Woman", "Oh, I'm not."),
            )
        )
        assert out == (
            "**Interviewer:**     What is she doing? "
            "How are you reading and walking?\n"
            "**Woman:**     Oh, I'm not."
        )

    def test_speaker_alternating_back_is_a_new_block(self):
        out = format_dialogue(
            lines(
                (1, 0.0, 1.0, "A", "one"),
                (2, 1.0, 2.0, "B", "two"),
                (3, 2.0, 3.0, "A", "three"),
            )
        )
        assert out.splitlines() == [
            "**A:**     one",
            "**B:**     two",
            "**A:**     three",
        ]

    def test_blank_text_is_skipped_and_blank_speaker_defaults(self):
        out = format_dialogue(
            [
                {"speaker": "", "text": "hello"},
                {"speaker": "A", "text": "   "},
            ]
        )
        assert out == "**Speaker:**     hello"

    def test_empty_input(self):
        assert format_dialogue([]) == ""
        assert format_dialogue(None) == ""


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


@pytest.fixture
def templates(monkeypatch):
    """Stand in for the templates app.project_types owns."""
    monkeypatch.setattr(
        project_types,
        "_LOCALISATION_SWAP_PROMPT",
        "SWAP from {source_language} to {target_language}\n{dialogue}",
        raising=False,
    )
    monkeypatch.setattr(
        project_types,
        "_LOCALISATION_KEEP_PROMPT",
        "KEEP from {source_language} to {target_language}\n{dialogue}",
        raising=False,
    )


class TestBuildPrompt:
    DIALOGUE = lines((1, 0.0, 1.0, "Woman", "こんにちは"))

    def test_swap_template_and_lowercase_language_names(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="ja",
            swap_character=True,
        )
        assert out == "SWAP from english to japanese\n**Woman:**     こんにちは"

    def test_keep_template(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="ja",
            swap_character=False,
        )
        assert out.startswith("KEEP from english to japanese")

    def test_unknown_detected_language_is_not_faked_as_english(self, templates):
        """languages.label_for() defaults to English; a detected code outside
        the five we deliver must NOT be printed as 'english'."""
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="fr",
            target_language="ja",
            swap_character=True,
        )
        assert out.startswith("SWAP from fr to japanese")

    def test_missing_template_raises(self, monkeypatch):
        monkeypatch.delattr(
            project_types, "_LOCALISATION_SWAP_PROMPT", raising=False
        )
        with pytest.raises(LocalisationError, match="_LOCALISATION_SWAP_PROMPT"):
            build_prompt(
                lines=self.DIALOGUE,
                source_language="en",
                target_language="ja",
                swap_character=True,
            )

    def test_unexpected_placeholder_raises(self, monkeypatch):
        monkeypatch.setattr(
            project_types,
            "_LOCALISATION_KEEP_PROMPT",
            "KEEP {dialogue} {nope}",
            raising=False,
        )
        with pytest.raises(LocalisationError, match="unexpected placeholder"):
            build_prompt(
                lines=self.DIALOGUE,
                source_language="en",
                target_language="ja",
                swap_character=False,
            )



class TestRealLocalisationSeedanceTemplates:
    """build_prompt against the live project_types templates (adaptive lips)."""

    DIALOGUE = lines((1, 0.0, 1.0, "Woman", "こんにちは"))

    @pytest.mark.parametrize("swap_character", [True, False])
    def test_renders_slots_and_adaptive_speech_rules(self, swap_character):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="ja",
            swap_character=swap_character,
        )
        assert "{source_language}" not in out
        assert "{target_language}" not in out
        assert "{dialogue}" not in out
        assert "(from english to japanese)" in out
        assert out.endswith("**Woman:**     こんにちは")
        lowered = out.lower()
        assert "lip-sync the translated speech" in lowered
        assert "facial expressions" in lowered
        assert "emotional reactions" in lowered
        assert "original motion and lip movements" not in lowered
        assert "do not freeze the original face or mouth motion" in lowered
        assert "on-screen text" in lowered
        if swap_character:
            assert "replace the main person" in lowered
            assert "reference image" in lowered
        else:
            assert "do not replace the character" in lowered
            assert "reference image" not in lowered


# ---------------------------------------------------------------------------
# Prompt contents — docs/localisation.md §4.3 is a checklist, so check it
# ---------------------------------------------------------------------------


class TestPromptRequirements:
    def test_versions(self):
        assert localisation.TRANSCRIBE_PROMPT_VERSION == "transcribe/v2"
        assert localisation.TRANSLATE_PROMPT_VERSION == "translate/v2"

    @pytest.mark.parametrize(
        "needle",
        [
            "never assume it is",          # detect, don't assume, the language
            "ISO 639-1",
            "VIDEO SUMMARY",
            "SCENE CONTEXT",
            "video_summary",
            "scene_context",
            "point at the person",         # visually grounded speaker label
            'NEVER use "Speaker 1"',
            '"on_screen": true',           # on/off camera flag
            "on_screen_text",              # captions kept separate
            "VERBATIM",
            "seconds from the beginning",  # per-line timestamps
            '"lines": []',                 # zero lines rather than inventing
            "NEVER invent",
        ],
    )
    def test_transcribe_prompt_covers_requirement(self, needle):
        assert needle in localisation.TRANSCRIBE_PROMPT

    @pytest.mark.parametrize(
        "needle",
        [
            "DURATION IN SECONDS",         # each line's duration is supplied
            "SAY OUT LOUD",                # spoken duration, not word count
            "Do NOT match the word count",
            "REWRITE, DO NOT TRANSLATE",   # free to rephrase
            "KEEP THE REGISTER",
            "LEAVE THESE UNTRANSLATED",    # brands, numbers, prices
            "Speechify",
            "NATIVE SCRIPT ONLY",
            "Never romanise",
            "ONE LINE IN, ONE LINE OUT",   # 1:1 with ids preserved
            "SPOKEN DIALOGUE ONLY",        # never translate on_screen_text
        ],
    )
    def test_translate_prompt_covers_requirement(self, needle):
        assert needle in localisation.TRANSLATE_PROMPT

    def test_translate_prompt_formats_without_stray_placeholders(self):
        out = localisation.TRANSLATE_PROMPT.format(
            source_language="english",
            target_language="japanese",
            lines_block="[1] speaker: A | duration: 1.0s\n    hi",
            video_summary="A short promo.",
            scene_context="Two people on a street.",
        )
        assert "{" not in out.replace('{"lines"', "").replace('{"id"', "")
        assert '{"lines": [{"id": 1, "text": "<translated line>"}]}' in out
