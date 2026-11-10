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
import os
import re

import httpx
import pytest
import respx

from app import kie_client, localisation, project_types
from app.kie_client import KieClient
from app.media import MediaError
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

CHAT_URL = f"{JOBS_BASE}/gemini-3.1-pro/v1/chat/completions"
RESPONSES_URL = f"{JOBS_BASE}/codex/v1/responses"
UPLOAD_URL = f"{UPLOAD_BASE}/api/file-stream-upload"
FALLBACK_CHAT_URL = f"{JOBS_BASE}/gemini-3-pro/v1/chat/completions"
FLASH_CHAT_URL = f"{JOBS_BASE}/gemini-3-6-flash-openai/v1/chat/completions"
FLASH_38_CHAT_URL = f"{JOBS_BASE}/gemini-3-8-flash-openai/v1/chat/completions"

# kie.ai's maintenance answer, verbatim from prod 2026-09-25: an HTTP 200.
MAINTENANCE = httpx.Response(
    200,
    json={
        "code": 500,
        "msg": "The server is currently being maintained, please try again later~",
    },
)


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
def no_retry_sleep(monkeypatch):
    """Skip tenacity's real backoff between retried LLM attempts."""
    monkeypatch.setattr(KieClient._chat_with_retry.retry, "sleep", lambda _s: None)


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


def upload_request() -> httpx.Request:
    """The upload call, which is never ``respx.calls.last`` — the chat call is."""
    for call in respx.calls:
        if str(call.request.url) == UPLOAD_URL:
            return call.request
    raise AssertionError("no upload request was made")


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
    "scene_context": "  Interviewer is off-camera; woman holds a phone labeled Listenify.  ",
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
        assert out["model"] == "gemini-3.1-pro"
        assert out["created_at"].endswith("+00:00")
        # "EN" is normalised to a bare lowercase ISO 639-1 code.
        assert out["source_language"] == "en"
        # Context strings are stripped and stored on the envelope.
        assert out["video_summary"] == "A street-interview promo for a reading app."
        assert out["scene_context"] == (
            "Interviewer is off-camera; woman holds a phone labeled Listenify."
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
        assert out["schema_version"] == 3
        assert out["prompt_version"] == "transcribe/v3"

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
    def test_unavailable_primary_falls_back_on_same_upload(
        self, kie, clip, no_retry_sleep
    ):
        """kie.ai's HTTP-200 "being maintained" 500 (gemini-2.5-pro, 2026-09-25)
        is retried, then the fallback model answers the same uploaded URL."""
        upload = respx.post(UPLOAD_URL).mock(
            return_value=httpx.Response(
                200,
                json={"code": 200, "data": {"downloadUrl": "https://tempfile.test/h.mp4"}},
            )
        )
        primary = respx.post(CHAT_URL).mock(return_value=MAINTENANCE)
        fallback = respx.post(FALLBACK_CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert primary.call_count == 4  # the retry budget, then give up
        assert fallback.call_count == 1
        assert upload.call_count == 1
        body = json.loads(fallback.calls[0].request.content)
        assert body["messages"][0]["content"][1]["image_url"]["url"] == (
            "https://tempfile.test/h.mp4"
        )
        assert out["model"] == "gemini-3-pro"
        assert len(out["lines"]) == 2

    @respx.mock
    def test_unsupported_model_falls_back_without_retry(self, kie, clip):
        mock_upload()
        primary = respx.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200, json={"code": 422, "msg": "The model is not supported"}
            )
        )
        respx.post(FALLBACK_CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert primary.call_count == 1
        assert out["model"] == "gemini-3-pro"

    @respx.mock
    def test_retired_model_404_falls_back_without_retry(self, kie, clip):
        """2026-10-13: kie.ai retired gemini-2.5-pro and answered every call
        with HTTP 200 {"code":404,"msg":"Invalid model name: HTTP 404"}."""
        mock_upload()
        primary = respx.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200, json={"code": 404, "msg": "Invalid model name: HTTP 404"}
            )
        )
        respx.post(FALLBACK_CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert primary.call_count == 1
        assert out["model"] == "gemini-3-pro"

    @respx.mock
    def test_second_fallback_when_first_is_down_too(
        self, kie, clip, no_retry_sleep
    ):
        """2026-09-25: 2.5-pro in maintenance, then 3.1-pro "internal error",
        while gemini-3-5-flash-openai still answered."""
        mock_upload()
        respx.post(CHAT_URL).mock(return_value=MAINTENANCE)
        respx.post(FALLBACK_CHAT_URL).mock(
            return_value=httpx.Response(
                200, json={"code": 500, "msg": "internal error, please try again later."}
            )
        )
        flash = respx.post(FLASH_CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert flash.call_count == 1
        assert out["model"] == "gemini-3-6-flash-openai"

    @respx.mock
    def test_all_unavailable_names_each(self, kie, clip, no_retry_sleep):
        mock_upload()
        for url in (CHAT_URL, FALLBACK_CHAT_URL, FLASH_CHAT_URL, FLASH_38_CHAT_URL):
            respx.post(url).mock(return_value=MAINTENANCE)

        with pytest.raises(LocalisationError) as exc_info:
            transcribe_video(clip, kie=kie)

        msg = str(exc_info.value)
        for model in (
            "gemini-3.1-pro", "gemini-3-pro",
            "gemini-3-6-flash-openai", "gemini-3-8-flash-openai",
        ):
            assert model in msg
        assert "being maintained" in msg

    @respx.mock
    def test_fallback_list_skips_blanks_and_the_primary(
        self, kie, clip, no_retry_sleep, monkeypatch
    ):
        monkeypatch.setattr(
            localisation.settings,
            "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS",
            " gemini-3.1-pro , ,gemini-3-6-flash-openai ",
        )
        mock_upload()
        primary = respx.post(CHAT_URL).mock(return_value=MAINTENANCE)
        respx.post(FLASH_CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert primary.call_count == 4  # one model's retry budget, not two
        assert out["model"] == "gemini-3-6-flash-openai"

    @respx.mock
    def test_auth_error_does_not_fall_back(self, kie, clip):
        """A bad key fails every model alike — don't bill a second call."""
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=httpx.Response(200, json={"code": 401, "msg": "Unauthorized"})
        )
        fallback = respx.post(FALLBACK_CHAT_URL)

        with pytest.raises(LocalisationError, match="code=401"):
            transcribe_video(clip, kie=kie)
        assert not fallback.called

    @respx.mock
    def test_fallback_disabled_by_empty_setting(
        self, kie, clip, no_retry_sleep, monkeypatch
    ):
        monkeypatch.setattr(
            localisation.settings, "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS", ""
        )
        mock_upload()
        respx.post(CHAT_URL).mock(return_value=MAINTENANCE)
        fallback = respx.post(FALLBACK_CHAT_URL)

        with pytest.raises(LocalisationError, match="code=500"):
            transcribe_video(clip, kie=kie)
        assert not fallback.called

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


class TestTranscribeUploadProxy:
    """What reaches kie.ai is a cheap re-encode, not the source.

    A delivery-quality source is both too big for the upload host (a 59s
    1080x1920 phone capture is 136 MB against a 100 MB ceiling) and pure waste
    against a per-input-token bill, and the model only has to *read* the video.
    """

    @respx.mock
    def test_uploads_the_proxy_not_the_source(self, kie, clip, monkeypatch, tmp_path):
        def fake_proxy(src, dst):
            with open(dst, "wb") as fh:
                fh.write(b"PROXYBYTES")
            return dst

        monkeypatch.setattr(localisation.media, "make_llm_proxy", fake_proxy)
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        transcribe_video(clip, kie=kie)

        body = upload_request().content
        assert b"PROXYBYTES" in body
        assert b"fakevideo" not in body

    @respx.mock
    def test_upload_keeps_the_source_file_name(self, kie, clip, monkeypatch):
        """kie.ai's file listing should name the source, not a throwaway temp file."""
        def fake_proxy(src, dst):
            with open(dst, "wb") as fh:
                fh.write(b"PROXYBYTES")
            return dst

        monkeypatch.setattr(localisation.media, "make_llm_proxy", fake_proxy)
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        transcribe_video(clip, kie=kie)

        body = upload_request().content.decode(errors="replace")
        assert "_hook.mp4" in body
        assert "proxy.mp4" not in body

    @respx.mock
    def test_same_named_sources_get_distinct_upload_names(self, kie, tmp_path):
        """The upload host overwrites same-named files in a folder. Two
        captions of two different runs (both "final.mp4") started together on
        2026-09-25 and BOTH got the second run's transcript."""
        clips = []
        for run in ("run_a", "run_b"):
            (tmp_path / run).mkdir()
            clip_path = tmp_path / run / "final.mp4"
            clip_path.write_bytes(b"fakevideo")
            clips.append(str(clip_path))
        upload = respx.post(UPLOAD_URL).mock(
            return_value=httpx.Response(
                200,
                json={"code": 200, "data": {"downloadUrl": "https://tempfile.test/h.mp4"}},
            )
        )
        respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        for c in clips:
            transcribe_video(c, kie=kie)

        names = [
            re.search(rb'name="fileName"\r\n\r\n([^\r]+)', call.request.content).group(1)
            for call in upload.calls
        ]
        assert len(names) == 2
        assert names[0] != names[1]
        assert all(n.endswith(b"_final.mp4") for n in names)

    @respx.mock
    def test_falls_back_to_the_source_when_the_proxy_fails(
        self, kie, clip, monkeypatch
    ):
        """A broken re-encode must not be what fails a transcription.

        The proxy is an optimisation on a file that may well have been small
        enough already; the upload's size guard judges the original on its own
        merits.
        """
        def boom(src, dst):
            raise MediaError("ffmpeg exploded")

        monkeypatch.setattr(localisation.media, "make_llm_proxy", boom)
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert out["source_language"] == "en"
        assert b"fakevideo" in upload_request().content

    @respx.mock
    def test_proxy_is_cleaned_up(self, kie, clip, monkeypatch):
        """The proxy is a temp file per transcription — it must not accumulate."""
        seen: list[str] = []

        def fake_proxy(src, dst):
            seen.append(dst)
            with open(dst, "wb") as fh:
                fh.write(b"PROXYBYTES")
            return dst

        monkeypatch.setattr(localisation.media, "make_llm_proxy", fake_proxy)
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        transcribe_video(clip, kie=kie)

        assert seen and not os.path.exists(seen[0])
        assert not os.path.exists(os.path.dirname(seen[0]))

    @respx.mock
    def test_proxy_is_cleaned_up_when_the_upload_fails(self, kie, clip, monkeypatch):
        """Cleanup must survive an exception raised *inside* the context body.

        The upload is the only thing the context wraps — the model call happens
        after it closes, deliberately, so a proxy is not held on disk for the
        minutes a transcription takes.
        """
        seen: list[str] = []

        def fake_proxy(src, dst):
            seen.append(dst)
            with open(dst, "wb") as fh:
                fh.write(b"PROXYBYTES")
            return dst

        monkeypatch.setattr(localisation.media, "make_llm_proxy", fake_proxy)
        respx.post(UPLOAD_URL).mock(return_value=httpx.Response(413, text="too big"))

        with pytest.raises(LocalisationError, match="upload failed"):
            transcribe_video(clip, kie=kie)

        assert seen and not os.path.exists(os.path.dirname(seen[0]))

    @respx.mock
    @pytest.mark.parametrize(
        "exc",
        [
            MediaError("ffmpeg exploded"),
            # probe() raises these on a malformed file — neither is a
            # MediaError, and both used to escape as-is.
            KeyError("width"),
            ValueError("could not convert string to float"),
            OSError("No space left on device"),
        ],
    )
    def test_any_proxy_failure_falls_back(self, kie, clip, monkeypatch, exc):
        """Building an optimisation must never be what fails a transcription."""
        def boom(src, dst):
            raise exc

        monkeypatch.setattr(localisation.media, "make_llm_proxy", boom)
        mock_upload()
        respx.post(CHAT_URL).mock(
            return_value=chat_response(json.dumps(TRANSCRIPT_JSON))
        )

        out = transcribe_video(clip, kie=kie)

        assert out["source_language"] == "en"
        assert b"fakevideo" in upload_request().content


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
        assert body["model"] == "gpt-6-1-sol"
        assert "response_format" not in body
        assert body["reasoning"] == {"effort": "high"}
        assert body["stream"] is False
        part = body["input"][0]["content"][0]
        assert part["type"] == "input_text"
        prompt = part["text"]
        assert "duration: 1.2s" in prompt
        assert "duration: 3.2s" in prompt
        assert "native japanese copywriter" in prompt
        assert "from english" in prompt
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
        """languages.label_for() defaults to English; a detected code we do not
        deliver must NOT be printed as 'english'."""
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="nl",
            target_language="ja",
            swap_character=True,
        )
        assert out.startswith("SWAP from nl to japanese")

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
        assert localisation.TRANSCRIBE_PROMPT_VERSION == "transcribe/v3"
        assert localisation.TRANSLATE_PROMPT_VERSION == "translate/v3"
        assert localisation.TRANSCRIPT_SCHEMA_VERSION == 3

    @pytest.mark.parametrize(
        "needle",
        [
            "never assume it is",          # detect, don't assume, the language
            "ISO 639-1",
            "VIDEO SUMMARY",               # high-level narrative/purpose
            "SCENE CONTEXT",               # translation-oriented situation
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
            "ADAPT, DO NOT TRANSLATE",     # free to rephrase
            "TARGET MARKET",               # es-MX, pt-BR, ...
            "trust the frames",            # frames beat a misheard word
            "FULL SOURCE TRANSCRIPT",
            "ON-SCREEN TEXT",
            "pause after",
            "Unit and",                    # "miles" -> "milhas", not kept
            "VIDEO SUMMARY:",              # context section labels
            "SCENE CONTEXT:",
            "KEEP THE REGISTER",
            "LEAVE THESE UNTRANSLATED",    # brands, numbers, prices
            "Listenify",
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
            market="Japan",
            lines_block="[1] speaker: A | duration: 1.0s\n    hi",
            full_transcript=">> [1] 0.0-1.0s A: hi",
            on_screen_text="(none)",
            video_summary="A short promo.",
            scene_context="Two people on a street.",
        )
        assert "{" not in out.replace('{"lines"', "").replace('{"id"', "")
        assert '{"lines": [{"id": 1, "text": "<adapted line>"}]}' in out
        assert "VIDEO SUMMARY:\nA short promo." in out
        assert "SCENE CONTEXT:\nTwo people on a street." in out


# ---------------------------------------------------------------------------
# Italian (it) — the sixth registry language through translate + build_prompt
# ---------------------------------------------------------------------------

# Realistic Italian: accents (è, ò, à), an apostrophe elision ("un po'") and
# the inverted-comma punctuation an LLM actually returns. These are the
# characters most likely to be mangled by an encoding or escaping bug between
# the provider response and the Seedance prompt.
IT_SRC = lines(
    (1, 0.0, 1.2, "off-screen interviewer", "What is she doing?"),
    (2, 1.6, 4.8, "the woman in the pattern shirt", "Oh, I'm not."),
)
IT_LINE_1 = "Perché è già così?"
IT_LINE_2 = "Un po' di più, però non troppo."


class TestTranslateIntoItalian:
    @respx.mock
    def test_request_names_italian_and_keeps_per_line_metadata(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps(
                    {"lines": [{"id": 2, "text": IT_LINE_2},
                               {"id": 1, "text": IT_LINE_1}]}
                )
            )
        )

        out = translate_lines(IT_SRC, source_language="en", target_language="it")

        # 1. The provider was asked for Italian by NAME, not by code.
        body = json.loads(route.calls[0].request.content)
        sent = json.dumps(body)
        assert "italian" in sent.lower()
        assert "japanese" not in sent.lower()

        # 2. Correspondence: original order, original ids, source text kept.
        assert [ln["id"] for ln in out] == [1, 2]
        assert out[0]["text"] == IT_LINE_1
        assert out[0]["source_text"] == "What is she doing?"
        assert out[1]["text"] == IT_LINE_2
        assert out[1]["source_text"] == "Oh, I'm not."

        # 3. Timing and speaker survive untouched.
        assert out[0]["speaker"] == "off-screen interviewer"
        assert (out[1]["start"], out[1]["end"]) == (1.6, 4.8)

        # 4. The accents and the elided apostrophe round-trip byte for byte.
        assert out[0]["text"] == "Perché è già così?"
        assert "po'" in out[1]["text"]

        # 5. The input list is not mutated.
        assert IT_SRC[0]["text"] == "What is she doing?"

    @respx.mock
    def test_italian_as_the_detected_source_is_named_too(self, shared_kie):
        """Detected source is unrestricted; Italian is simply one of the codes
        the registry can also name."""
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps({"lines": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}]})
            )
        )

        translate_lines(IT_SRC, source_language="it", target_language="en")

        sent = json.dumps(json.loads(route.calls[0].request.content)).lower()
        assert "italian" in sent
        assert "english" in sent


class TestBuildPromptItalian:
    DIALOGUE = lines((1, 0.0, 1.0, "Woman", IT_LINE_1))

    def test_swap_template_names_italian(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="it",
            swap_character=True,
        )
        assert out == f"SWAP from english to italian\n**Woman:**     {IT_LINE_1}"

    def test_keep_template_names_italian(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="it",
            swap_character=False,
        )
        assert out.startswith("KEEP from english to italian")
        assert IT_LINE_1 in out

    def test_italian_source_is_named_not_defaulted_to_english(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="it",
            target_language="es",
            swap_character=True,
        )
        assert out.startswith("SWAP from italian to spanish")

    def test_accented_dialogue_reaches_the_prompt_unmangled(self, templates):
        dialogue = lines(
            (1, 0.0, 1.0, "Woman", IT_LINE_1),
            (2, 1.0, 2.0, "Man", IT_LINE_2),
        )
        out = build_prompt(
            lines=dialogue,
            source_language="en",
            target_language="it",
            swap_character=False,
        )
        assert "Perché è già così?" in out
        assert "Un po' di più, però non troppo." in out


# ---------------------------------------------------------------------------
# Polish (pl) — the seventh registry language through translate + build_prompt
# ---------------------------------------------------------------------------

# Every Polish diacritic (ą ć ę ł ń ó ś ź ż) appears across these two lines —
# the characters an encoding or escaping bug between the provider response and
# the Seedance prompt would mangle.
PL_LINE_1 = "Cześć, co ona robi?"
PL_LINE_2 = "Źle, że łąka już zżółkła — ań, ę."


def test_language_name_for_polish():
    assert localisation._language_name("pl") == "polish"
    assert localisation._language_name("  PL ") == "polish"


class TestTranslateIntoPolish:
    @respx.mock
    def test_request_names_polish_and_round_trips_diacritics(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps(
                    {"lines": [{"id": 2, "text": PL_LINE_2},
                               {"id": 1, "text": PL_LINE_1}]},
                    ensure_ascii=False,
                )
            )
        )

        out = translate_lines(IT_SRC, source_language="en", target_language="pl")

        # The translation request names Polish — by NAME, not by code.
        sent = json.dumps(json.loads(route.calls[0].request.content)).lower()
        assert "polish" in sent
        assert "italian" not in sent

        assert [ln["id"] for ln in out] == [1, 2]
        assert out[0]["text"] == PL_LINE_1
        assert out[1]["text"] == PL_LINE_2
        assert out[0]["source_text"] == "What is she doing?"


class TestBuildPromptPolish:
    """The Seedance prompt names Polish in both assembly modes."""

    DIALOGUE = lines((1, 0.0, 1.0, "Woman", PL_LINE_1))

    def test_swap_template_names_polish(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="pl",
            swap_character=True,
        )
        assert out == f"SWAP from english to polish\n**Woman:**     {PL_LINE_1}"

    def test_keep_template_names_polish(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="pl",
            swap_character=False,
        )
        assert out.startswith("KEEP from english to polish")
        assert PL_LINE_1 in out

    def test_diacritics_reach_the_prompt_unmangled(self, templates):
        out = build_prompt(
            lines=lines((1, 0.0, 1.0, "Man", PL_LINE_2)),
            source_language="en",
            target_language="pl",
            swap_character=False,
        )
        assert PL_LINE_2 in out


# ---------------------------------------------------------------------------
# French (fr) — the eighth registry language through translate + build_prompt
# ---------------------------------------------------------------------------

# French accents, the cedilla, the œ ligature and guillemets (é è ê à ç ô û ï
# œ « ») appear across these two lines — the characters an encoding or escaping
# bug between the provider response and the Seedance prompt would mangle.
FR_LINE_1 = "Qu'est-ce qu'elle fait ?"
FR_LINE_2 = "Ça a été déjà vu à Noël — « l'œuvre » d'un être naïf, sûr."


def test_language_name_for_french():
    assert localisation._language_name("fr") == "french"
    assert localisation._language_name("  FR ") == "french"


class TestTranslateIntoFrench:
    @respx.mock
    def test_request_names_french_and_round_trips_diacritics(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(
                json.dumps(
                    {"lines": [{"id": 2, "text": FR_LINE_2},
                               {"id": 1, "text": FR_LINE_1}]},
                    ensure_ascii=False,
                )
            )
        )

        out = translate_lines(IT_SRC, source_language="en", target_language="fr")

        # The translation request names French — by NAME, not by code.
        sent = json.dumps(json.loads(route.calls[0].request.content)).lower()
        assert "french" in sent
        assert "italian" not in sent

        assert [ln["id"] for ln in out] == [1, 2]
        assert out[0]["text"] == FR_LINE_1
        assert out[1]["text"] == FR_LINE_2
        assert out[0]["source_text"] == "What is she doing?"


class TestBuildPromptFrench:
    """The Seedance prompt names French in both assembly modes."""

    DIALOGUE = lines((1, 0.0, 1.0, "Woman", FR_LINE_1))

    def test_swap_template_names_french(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="fr",
            swap_character=True,
        )
        assert out == f"SWAP from english to french\n**Woman:**     {FR_LINE_1}"

    def test_keep_template_names_french(self, templates):
        out = build_prompt(
            lines=self.DIALOGUE,
            source_language="en",
            target_language="fr",
            swap_character=False,
        )
        assert out.startswith("KEEP from english to french")
        assert FR_LINE_1 in out

    def test_diacritics_reach_the_prompt_unmangled(self, templates):
        out = build_prompt(
            lines=lines((1, 0.0, 1.0, "Man", FR_LINE_2)),
            source_language="en",
            target_language="fr",
            swap_character=False,
        )
        assert FR_LINE_2 in out


# ---------------------------------------------------------------------------
# translate/v3 — frames, full transcript, market, model fallback (2026-10-20)
# ---------------------------------------------------------------------------


V3_ALL = lines(
    (1, 0.0, 1.2, "off-screen interviewer", "What is she doing?"),
    (2, 1.6, 4.8, "the woman in the pattern shirt", "Oh, I'm not."),
    (3, 12.0, 14.0, "the woman in the pattern shirt", "Download it today."),
)
V3_HOOK = V3_ALL[:2]
TWO_LINES = json.dumps({"lines": [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]})
SOL_UNSUPPORTED = httpx.Response(
    200, json={"code": 422, "msg": "The model is not supported"}
)


class TestTranslateV3Context:
    @respx.mock
    def test_frames_are_sent_as_captioned_input_images(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(TWO_LINES)
        )

        translate_lines(
            V3_HOOK,
            source_language="en",
            target_language="es",
            frames=[(0.3, b"\xff\xd8one"), (2.4, b"\xff\xd8two")],
        )

        content = json.loads(route.calls[0].request.content)["input"][0]["content"]
        assert [p["type"] for p in content] == [
            "input_text", "input_text", "input_image", "input_text", "input_image",
        ]
        assert content[1]["text"] == "Frame at 0.3s:"
        assert content[3]["text"] == "Frame at 2.4s:"
        assert content[2]["image_url"].startswith("data:image/jpeg;base64,")
        assert content[4]["image_url"] == "data:image/jpeg;base64,/9h0d28="

    @respx.mock
    def test_prompt_carries_full_transcript_on_screen_text_and_market(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(TWO_LINES)
        )

        translate_lines(
            V3_HOOK,
            source_language="en",
            target_language="es",
            context_lines=V3_ALL,
            on_screen_text=[{"start": 0.0, "end": 3.0, "text": "SALE 50%"}],
        )

        prompt = json.loads(route.calls[0].request.content)["input"][0]["content"][0][
            "text"
        ]
        assert ">> [1] 0.0-1.2s off-screen interviewer: What is she doing?" in prompt
        assert ">> [2] 1.6-4.8s" in prompt
        # Context-only line: in the transcript, unmarked, never in the hook block.
        assert "   [3] 12.0-14.0s the woman in the pattern shirt: Download it today." in prompt
        assert "0.0-3.0s: SALE 50%" in prompt
        assert "Mexico" in prompt and "celular" in prompt
        # Pause runs to the next line of the WHOLE transcript.
        assert (
            "[1] speaker: off-screen interviewer (visible) | 0.0-1.2s | "
            "duration: 1.2s | about 17 characters, max 21 | pause after: 0.4s"
        ) in prompt
        assert "duration: 3.2s | about 45 characters, max 56 | pause after: 7.2s" in prompt

    @respx.mock
    def test_pt_targets_brazil(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(TWO_LINES)
        )
        translate_lines(V3_HOOK, source_language="en", target_language="pt")
        prompt = json.loads(route.calls[0].request.content)["input"][0]["content"][0][
            "text"
        ]
        assert "TARGET MARKET: Brazil (Brazilian Portuguese" in prompt

    @respx.mock
    def test_unavailable_model_falls_back_to_luna(self, shared_kie, no_retry_sleep):
        calls = []

        def _answer(request):
            model = json.loads(request.content)["model"]
            calls.append(model)
            if model == "gpt-6-1-sol":
                return SOL_UNSUPPORTED
            return responses_response(TWO_LINES)

        respx.post(RESPONSES_URL).mock(side_effect=_answer)

        out = translate_lines(V3_HOOK, source_language="en", target_language="ja")

        assert calls == ["gpt-6-1-sol", "gpt-5-6-luna"]
        assert [ln["text"] for ln in out] == ["a", "b"]

    @respx.mock
    def test_every_model_unavailable_raises(self, shared_kie, no_retry_sleep):
        respx.post(RESPONSES_URL).mock(return_value=SOL_UNSUPPORTED)
        with pytest.raises(LocalisationError, match="then"):
            translate_lines(V3_HOOK, source_language="en", target_language="ja")

    @respx.mock
    def test_fallback_disabled_by_empty_setting(
        self, shared_kie, no_retry_sleep, monkeypatch
    ):
        monkeypatch.setattr(
            localisation.settings, "LOCALISATION_TRANSLATE_FALLBACK_MODELS", ""
        )
        route = respx.post(RESPONSES_URL).mock(return_value=SOL_UNSUPPORTED)
        with pytest.raises(LocalisationError):
            translate_lines(V3_HOOK, source_language="en", target_language="ja")
        assert route.call_count == 1

    @respx.mock
    def test_broken_answer_is_not_a_reason_to_fall_back(self, shared_kie):
        """A 1:1 break is re-asked on the SAME model, never handed to luna."""
        calls = []

        def _answer(request):
            calls.append(json.loads(request.content)["model"])
            return responses_response(json.dumps({"lines": [{"id": 1, "text": "a"}]}))

        respx.post(RESPONSES_URL).mock(side_effect=_answer)
        with pytest.raises(LocalisationError, match="still wrong after 2 attempts"):
            translate_lines(V3_HOOK, source_language="en", target_language="ja")
        assert calls == ["gpt-6-1-sol", "gpt-6-1-sol"]


class TestHookFrameTimes:
    def test_start_plus_line_midpoints_inside_the_hook(self):
        assert localisation.hook_frame_times(V3_ALL, 10.0, max_frames=8) == [
            0.3, 0.6, 3.2,
        ]

    def test_thinned_evenly_not_truncated(self):
        many = lines(*[(i, float(i), float(i) + 0.5, "s", "x") for i in range(1, 20)])
        out = localisation.hook_frame_times(many, 20.0, max_frames=4)
        assert len(out) == 4
        assert out[0] == 0.3 and out[-1] == 19.25

    def test_disabled(self):
        assert localisation.hook_frame_times(V3_ALL, 10.0, max_frames=0) == []
        assert localisation.hook_frame_times(V3_ALL, 0.0, max_frames=8) == []

    def test_short_hook_start_frame_stays_inside(self):
        assert localisation.hook_frame_times([], 0.4, max_frames=8) == [0.2]


@respx.mock
def test_translate_uses_its_own_timeout(shared_kie, monkeypatch):
    seen = {}
    real = shared_kie.create_response

    def _spy(**kw):
        seen["timeout"] = kw["timeout_sec"]
        return real(**kw)

    monkeypatch.setattr(shared_kie, "create_response", _spy)
    respx.post(RESPONSES_URL).mock(return_value=responses_response(TWO_LINES))
    translate_lines(V3_HOOK, source_language="en", target_language="ja")
    assert seen["timeout"] == localisation.settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC == 90


def test_slow_unavailable_primary_does_not_fall_back(shared_kie, monkeypatch):
    """A primary that took minutes to be declared unavailable is not followed
    by the fallback — the endpoint would outlive nginx's 300s budget."""
    now = [0.0]
    called = []

    def _create_response(**kw):
        called.append(kw["model"])
        now[0] += 500.0
        raise kie_client.KieModelUnavailableError("down", retryable=True)

    monkeypatch.setattr(shared_kie, "create_response", _create_response)
    monkeypatch.setattr(localisation, "time", type("T", (), {"monotonic": staticmethod(lambda: now[0])}))
    with pytest.raises(LocalisationError, match="down"):
        translate_lines(V3_HOOK, source_language="en", target_language="ja")
    assert called == ["gpt-6-1-sol"]


# ---------------------------------------------------------------------------
# Speaking windows, character budgets and the shorten pass (2026-10-21)
# ---------------------------------------------------------------------------


class TestSpeakingWindow:
    def test_untouched_line(self):
        assert localisation.speaking_window(
            {"start": 1.0, "end": 2.0}, cuts=[5.0], hook_end=10.0
        ) == (1.0, 2.0)

    def test_clipped_by_the_hook_end(self):
        # ab35eba1 line 10: 10.0-11.5 under a 10.5s hook.
        assert localisation.speaking_window(
            {"start": 10.0, "end": 11.5}, hook_end=10.5
        ) == (10.0, 10.5)

    def test_seam_keeps_the_midpoint_side(self):
        assert localisation.speaking_window(
            {"start": 4.0, "end": 5.5}, cuts=[5.0]
        ) == (4.0, 5.0)
        assert localisation.speaking_window(
            {"start": 4.8, "end": 6.0}, cuts=[5.0]
        ) == (5.0, 6.0)


FIT_HOOK = lines(
    (1, 0.0, 1.0, "the woman", "Show me your book."),
    (2, 10.0, 11.5, "the phone voice", "As I set out"),
)


class TestFitToWindow:
    @respx.mock
    def test_prompt_carries_clipped_window_and_char_budget(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(TWO_LINES)
        )
        translate_lines(
            FIT_HOOK, source_language="en", target_language="fr", hook_end=10.5
        )
        prompt = json.loads(route.calls[0].request.content)["input"][0]["content"][0][
            "text"
        ]
        assert (
            "[2] speaker: the phone voice (visible) | 10.0-10.5s | duration: 0.5s "
            "| about 7 characters, max 8 | pause after: none (the clip ends here)"
        ) in prompt
        assert "the original runs 10.0-11.5s but the clip is cut at 10.5s" in prompt
        assert "NEVER exceed the max" in prompt

    @respx.mock
    def test_overrunning_line_is_shortened_others_kept(self, shared_kie):
        first = json.dumps({"lines": [
            {"id": 1, "text": "Montre-moi ton livre, s'il te plaît, vas-y."},
            {"id": 2, "text": "Alors"},
        ]})
        second = json.dumps({"lines": [
            {"id": 1, "text": "Ton livre ?"},
            {"id": 2, "text": "Alors que je partais"},  # longer: must be ignored
        ]})
        route = respx.post(RESPONSES_URL).mock(side_effect=[
            responses_response(first), responses_response(second),
        ])

        out = translate_lines(
            FIT_HOOK, source_language="en", target_language="fr", hook_end=10.5
        )

        assert route.call_count == 2
        assert [ln["text"] for ln in out] == ["Ton livre ?", "Alors"]
        shorten_prompt = json.loads(route.calls[1].request.content)["input"][0][
            "content"
        ][0]["text"]
        assert "YOUR PREVIOUS ANSWER:" in shorten_prompt
        assert "Never drop or replace a person's name" in shorten_prompt
        assert "[1] 1.0s window, max 17 characters" in shorten_prompt
        assert "[2]" not in shorten_prompt.split("These lines are too long")[1]

    @respx.mock
    def test_no_shorten_call_when_everything_fits(self, shared_kie):
        route = respx.post(RESPONSES_URL).mock(
            return_value=responses_response(json.dumps({"lines": [
                {"id": 1, "text": "Ton livre ?"}, {"id": 2, "text": "Alors"},
            ]}))
        )
        translate_lines(
            FIT_HOOK, source_language="en", target_language="fr", hook_end=10.5
        )
        assert route.call_count == 1

    @respx.mock
    def test_broken_shorten_answer_keeps_the_first(self, shared_kie):
        first = json.dumps({"lines": [
            {"id": 1, "text": "Montre-moi ton livre, s'il te plaît, vas-y."},
            {"id": 2, "text": "Alors"},
        ]})
        respx.post(RESPONSES_URL).mock(side_effect=[
            responses_response(first),
            responses_response(json.dumps({"lines": [{"id": 1, "text": "x"}]})),
        ])
        out = translate_lines(
            FIT_HOOK, source_language="en", target_language="fr", hook_end=10.5
        )
        assert out[0]["text"] == "Montre-moi ton livre, s'il te plaît, vas-y."


@respx.mock
def test_shorten_pass_never_drops_a_name(shared_kie):
    """Terra (AICP-248 follow-up): shorter is not enough — a rewrite that loses
    a kept name/brand/number is refused and the first wording stays."""
    hook = lines((1, 0.0, 0.3, "off-camera speaker", "Sophia?"))
    respx.post(RESPONSES_URL).mock(side_effect=[
        responses_response(json.dumps({"lines": [{"id": 1, "text": "¿Sophia?"}]})),
        responses_response(json.dumps({"lines": [{"id": 1, "text": "Oye."}]})),
    ])
    out = translate_lines(hook, source_language="en", target_language="es")
    assert out[0]["text"] == "¿Sophia?"


def test_protected_tokens():
    assert localisation._protected_tokens(
        "Get Listenify for 50% off, Sophia", "Consigue Listenify con 50% menos, Sophia"
    ) == {"Listenify", "50", "Sophia"}
    assert localisation._protected_tokens("No, what is that?", "No, ¿qué es eso?") == set()
    assert localisation._protected_tokens(
        "Sell it on eBay with your iPhone", "Véndelo en eBay con tu iPhone"
    ) == {"eBay", "iPhone"}
    assert localisation._protected_tokens(
        "Visit x.co today", "Entra a x.co hoy"
    ) == {"x.co"}


@respx.mock
def test_slow_first_answer_skips_the_shorten_pass(shared_kie, monkeypatch):
    now = [0.0]
    real = shared_kie.create_response

    def _slow(**kw):
        now[0] += 61.0
        return real(**kw)

    monkeypatch.setattr(shared_kie, "create_response", _slow)
    monkeypatch.setattr(
        localisation, "time",
        type("T", (), {"monotonic": staticmethod(lambda: now[0])}),
    )
    route = respx.post(RESPONSES_URL).mock(return_value=responses_response(
        json.dumps({"lines": [
            {"id": 1, "text": "Montre-moi ton livre, s'il te plaît, vas-y."},
            {"id": 2, "text": "Alors"},
        ]})
    ))
    translate_lines(FIT_HOOK, source_language="en", target_language="fr", hook_end=10.5)
    assert route.call_count == 1
