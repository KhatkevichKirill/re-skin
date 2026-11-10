"""
Tests for KieClient — no real network calls, no real API key.

Uses respx to intercept httpx requests.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from unittest.mock import patch

import httpx
import pytest
import respx

from app import kie_client
from app.kie_client import (
    KieChatError,
    KieClient,
    KieModelUnavailableError,
    KieTaskError,
    KieTaskFailed,
    KieUploadError,
    get_shared_client,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FAKE_KEY = "test-api-key-not-real"
UPLOAD_BASE = "https://upload.test"
JOBS_BASE = "https://jobs.test"


def _make_client(sleep_calls: list | None = None) -> KieClient:
    """Return a KieClient pointed at test bases with a no-op sleep."""
    sleep_log: list = [] if sleep_calls is None else sleep_calls

    def _fake_sleep(secs: float) -> None:
        sleep_log.append(secs)

    return KieClient(
        api_key=FAKE_KEY,
        upload_base=UPLOAD_BASE,
        jobs_base=JOBS_BASE,
        sleep_fn=_fake_sleep,
    )


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------


class TestUploadFile:
    @respx.mock
    def test_returns_download_url(self, tmp_path):
        """upload_file posts multipart and returns the downloadUrl."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"fakevideo")

        upload_url = f"{UPLOAD_BASE}/api/file-stream-upload"
        respx.post(upload_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "code": 200,
                    "data": {
                        "downloadUrl": "https://tempfile.test/clip.mp4",
                        "fileName": "clip.mp4",
                        "fileSize": 9,
                    },
                },
            )
        )

        client = _make_client()
        url = client.upload_file(str(src), upload_path="charswap/segments")
        assert url == "https://tempfile.test/clip.mp4"

        # Verify the request contained the expected form fields
        req = respx.calls.last.request
        body = req.content.decode(errors="replace")
        assert "charswap/segments" in body
        assert "clip.mp4" in body

    @respx.mock
    def test_raises_on_non_200(self, tmp_path):
        """upload_file raises KieUploadError on a 4xx response."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"fakevideo")

        respx.post(f"{UPLOAD_BASE}/api/file-stream-upload").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        client = _make_client()
        with pytest.raises(KieUploadError, match="401"):
            client.upload_file(str(src))

    @respx.mock
    def test_raises_when_download_url_missing(self, tmp_path):
        """upload_file raises KieUploadError if downloadUrl absent from response."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"fakevideo")

        respx.post(f"{UPLOAD_BASE}/api/file-stream-upload").mock(
            return_value=httpx.Response(200, json={"success": True, "data": {}})
        )

        client = _make_client()
        with pytest.raises(KieUploadError, match="downloadUrl"):
            client.upload_file(str(src))


class TestUploadSizeGuard:
    """The host refuses an oversized body with a bare 500 (kie_client._UPLOAD_MAX_BYTES).

    That 500 is classed transient, so without a pre-flight check a too-large
    file costs four attempts and is reported as a server error — pointing the
    operator at kie.ai instead of at their file.
    """

    @respx.mock
    def test_oversized_file_is_refused_without_a_request(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kie_client, "_UPLOAD_MAX_BYTES", 1_000)
        src = tmp_path / "huge.mp4"
        src.write_bytes(b"x" * 2_000)

        route = respx.post(f"{UPLOAD_BASE}/api/file-stream-upload").mock(
            return_value=httpx.Response(500)
        )

        client = _make_client()
        with pytest.raises(KieUploadError) as excinfo:
            client.upload_file(str(src))

        # The network is never touched: the point is to not spend four
        # attempts on a deterministic refusal.
        assert not route.called
        assert "huge.mp4" in str(excinfo.value)

    @respx.mock
    def test_message_names_both_sizes(self, tmp_path, monkeypatch):
        """The operator needs to know how far over the limit they are."""
        monkeypatch.setattr(kie_client, "_UPLOAD_MAX_BYTES", 100_000_000)
        src = tmp_path / "huge.mp4"
        src.write_bytes(b"x" * 136_500_000)

        client = _make_client()
        with pytest.raises(KieUploadError) as excinfo:
            client.upload_file(str(src))

        message = str(excinfo.value)
        assert "136.5 MB" in message
        assert "100 MB" in message

    @respx.mock
    def test_file_at_the_limit_is_uploaded(self, tmp_path, monkeypatch):
        """The check is >, not >= — the limit itself is a legal size."""
        monkeypatch.setattr(kie_client, "_UPLOAD_MAX_BYTES", 1_000)
        src = tmp_path / "exact.mp4"
        src.write_bytes(b"x" * 1_000)

        respx.post(f"{UPLOAD_BASE}/api/file-stream-upload").mock(
            return_value=httpx.Response(
                200, json={"data": {"downloadUrl": "https://tempfile.test/exact.mp4"}}
            )
        )

        client = _make_client()
        assert client.upload_file(str(src)) == "https://tempfile.test/exact.mp4"

    @respx.mock
    def test_zero_disables_the_check(self, tmp_path, monkeypatch):
        """KIE_UPLOAD_MAX_SIZE_MB=0 is the escape hatch if kie.ai raises the cap."""
        monkeypatch.setattr(kie_client, "_UPLOAD_MAX_BYTES", 0)
        src = tmp_path / "huge.mp4"
        src.write_bytes(b"x" * 2_000)

        respx.post(f"{UPLOAD_BASE}/api/file-stream-upload").mock(
            return_value=httpx.Response(
                200, json={"data": {"downloadUrl": "https://tempfile.test/huge.mp4"}}
            )
        )

        client = _make_client()
        assert client.upload_file(str(src)) == "https://tempfile.test/huge.mp4"

    def test_missing_file_reports_the_path(self, tmp_path):
        client = _make_client()
        with pytest.raises(KieUploadError, match="cannot read"):
            client.upload_file(str(tmp_path / "gone.mp4"))

    @respx.mock
    def test_5xx_body_is_logged(self, tmp_path, caplog):
        """Without this the host's own explanation of a 5xx is never recorded."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"fakevideo")

        respx.post(f"{UPLOAD_BASE}/api/file-stream-upload").mock(
            return_value=httpx.Response(500, text="file too large")
        )

        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="app.kie_client"):
            with pytest.raises(Exception):
                client.upload_file(str(src))

        assert "file too large" in caplog.text


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


class TestCreateTask:
    @respx.mock
    def test_returns_task_id_and_sends_correct_body(self):
        """create_task sends the right JSON body and returns taskId."""
        create_url = f"{JOBS_BASE}/api/v1/jobs/createTask"
        respx.post(create_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "success",
                    "data": {"taskId": "task-abc123", "recordId": "rec-xyz"},
                },
            )
        )

        client = _make_client()
        task_id = client.create_task(
            prompt="swap the character",
            reference_image_urls=["https://img.test/face.jpg"],
            reference_video_urls=["https://vid.test/seg.mp4"],
            resolution="480p",
            aspect_ratio="9:16",
            duration=9,
        )
        assert task_id == "task-abc123"

        # Inspect the request body
        req = respx.calls.last.request
        body = json.loads(req.content)
        assert body["model"] == "bytedance/seedance-2"
        inp = body["input"]
        assert inp["prompt"] == "swap the character"
        assert inp["reference_image_urls"] == ["https://img.test/face.jpg"]
        assert inp["reference_video_urls"] == ["https://vid.test/seg.mp4"]
        assert inp["resolution"] == "480p"
        assert inp["aspect_ratio"] == "9:16"
        assert inp["duration"] == 9

    def test_duration_too_low_raises(self):
        """create_task raises ValueError if duration < 4."""
        client = _make_client()
        with pytest.raises(ValueError, match="duration must be between"):
            client.create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=3,
            )

    def test_duration_too_high_raises(self):
        """create_task raises ValueError if duration > 15."""
        client = _make_client()
        with pytest.raises(ValueError, match="duration must be between"):
            client.create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=20,
            )

    @respx.mock
    def test_raises_on_http_error(self):
        """create_task raises KieTaskError on a 4xx response."""
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(403, json={"error": "Forbidden"})
        )
        client = _make_client()
        with pytest.raises(KieTaskError, match="403"):
            client.create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=9,
            )

    @respx.mock
    @pytest.mark.parametrize(
        "model",
        ["bytedance/seedance-2-fast", "bytedance/seedance-2-mini"],
    )
    def test_sends_given_model_id(self, model):
        """create_task forwards the *model* kwarg verbatim in the payload."""
        create_url = f"{JOBS_BASE}/api/v1/jobs/createTask"
        respx.post(create_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "success",
                    "data": {"taskId": "task-variant", "recordId": "rec-variant"},
                },
            )
        )

        client = _make_client()
        task_id = client.create_task(
            prompt="swap the character",
            reference_image_urls=["https://img.test/face.jpg"],
            reference_video_urls=["https://vid.test/seg.mp4"],
            duration=9,
            model=model,
        )
        assert task_id == "task-variant"

        req = respx.calls.last.request
        body = json.loads(req.content)
        assert body["model"] == model

    def test_model_defaults_to_base_seedance(self):
        """Omitting *model* keeps the existing bytedance/seedance-2 default."""
        import inspect

        from app.kie_client import KieClient

        sig = inspect.signature(KieClient.create_task)
        assert sig.parameters["model"].default == "bytedance/seedance-2"


# ---------------------------------------------------------------------------
# create_task — per-model duration bounds (Seedance 2.5 goes to 30s)
# ---------------------------------------------------------------------------


class TestCreateTaskDurationPerModel:
    """The accepted duration range is looked up per *model*, not hardcoded.

    create_task's contract is the kie.ai model id, so the client maps that back
    to the app.ai_models registry entry to validate against: [4, 15] for the
    Seedance 2.0 family, [4, 30] for Seedance 2.5.
    """

    @staticmethod
    def _mock_create(task_id: str = "task-2-5"):
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200,
                json={"code": 200, "msg": "success", "data": {"taskId": task_id}},
            )
        )

    @respx.mock
    def test_2_5_requires_the_follow_input_sentinel(self):
        """Seedance 2.5 takes only -1: it reads the length off the input video.

        Learned from the live API, which 500s on a concrete value:
          `ratio` must be `adaptive`. `duration` must be -1.
        """
        self._mock_create()
        client = _make_client()
        task_id = client.create_task(
            prompt="swap the character",
            reference_image_urls=[],
            reference_video_urls=["https://vid.test/seg.mp4"],
            duration=-1,
            model="bytedance/seedance-2-5",
        )
        assert task_id == "task-2-5"
        body = json.loads(respx.calls.last.request.content)
        assert body["input"]["duration"] == -1

    @pytest.mark.parametrize("duration", [4, 22, 30, 31])
    def test_2_5_rejects_concrete_durations(self, duration):
        """Even a length inside 2.5's nominal 4-30s window is refused."""
        client = _make_client()
        with pytest.raises(ValueError, match="duration must be -1"):
            client.create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=duration,
                model="bytedance/seedance-2-5",
            )

    @respx.mock
    def test_the_two_generations_want_opposite_values(self):
        """-1 for 2.5, a real length for 2.0 — swapped, both are rejected."""
        self._mock_create()
        client = _make_client()

        client.create_task(
            prompt="x", reference_image_urls=[], reference_video_urls=[],
            duration=-1, model="bytedance/seedance-2-5",
        )
        client.create_task(
            prompt="x", reference_image_urls=[], reference_video_urls=[],
            duration=12, model="bytedance/seedance-2",
        )

        with pytest.raises(ValueError, match="duration must be -1"):
            client.create_task(
                prompt="x", reference_image_urls=[], reference_video_urls=[],
                duration=12, model="bytedance/seedance-2-5",
            )
        with pytest.raises(ValueError, match="duration must be between 4 and 15"):
            client.create_task(
                prompt="x", reference_image_urls=[], reference_video_urls=[],
                duration=-1, model="bytedance/seedance-2",
            )

    def test_unknown_model_id_falls_back_to_the_2_0_bound(self):
        """An id absent from the registry keeps the historical 4-15s rule."""
        client = _make_client()
        with pytest.raises(ValueError, match="duration must be between 4 and 15"):
            client.create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=22,
                model="bytedance/seedance-99-turbo",
            )


# ---------------------------------------------------------------------------
# create_task — the Seedance 2.5 `generate_audio` switch
# ---------------------------------------------------------------------------


class TestCreateTaskGenerateAudio:
    """`generate_audio` is only put on the wire when the caller asks for it.

    The 2.0 family rejects the field outright, so "not passed" must mean the key
    is missing from the JSON body — present-and-false is a different request.
    """

    @staticmethod
    def _mock_create():
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200,
                json={"code": 200, "msg": "success", "data": {"taskId": "t-audio"}},
            )
        )

    def _create(self, client, **kw) -> dict:
        # Duration is validated per model, and Seedance 2.5 accepts only -1
        # (it reads the length off the input video). Derive it from the registry
        # so this helper stays correct for whichever model a test names.
        from app import ai_models
        model_id = kw.get("model", "bytedance/seedance-2")
        key = next(
            (k for k, sp in ai_models.AI_MODELS.items()
             if sp.kie_model_id == model_id),
            "seedance",
        )
        kw.setdefault("duration", ai_models.duration_for(key, 0.0, 8.0))
        client.create_task(
            prompt="swap the character",
            reference_image_urls=[],
            reference_video_urls=["https://vid.test/seg.mp4"],
            **kw,
        )
        return json.loads(respx.calls.last.request.content)["input"]

    @respx.mock
    @pytest.mark.parametrize("value", [True, False])
    def test_included_in_input_when_passed(self, value):
        self._mock_create()
        inp = self._create(
            _make_client(), model="bytedance/seedance-2-5", generate_audio=value
        )
        assert inp["generate_audio"] is value

    @respx.mock
    def test_key_absent_when_not_passed(self):
        """Absent — not merely falsy. This is what keeps 2.0 payloads unchanged."""
        self._mock_create()
        inp = self._create(_make_client(), model="bytedance/seedance-2")
        assert "generate_audio" not in inp

    @respx.mock
    def test_explicit_none_is_also_omitted(self):
        """None is the "omit it" signal, matching the parameter's default."""
        self._mock_create()
        inp = self._create(
            _make_client(), model="bytedance/seedance-2-5", generate_audio=None
        )
        assert "generate_audio" not in inp

    def test_default_is_none(self):
        """The signature's default documents the omit-by-default contract."""
        import inspect

        from app.kie_client import KieClient

        sig = inspect.signature(KieClient.create_task)
        assert sig.parameters["generate_audio"].default is None

    @respx.mock
    def test_rest_of_the_payload_is_unchanged_by_the_switch(self):
        """Adding the flag must not disturb any other field."""
        self._mock_create()
        client = _make_client()
        without = self._create(client, model="bytedance/seedance-2-5")
        with_flag = self._create(
            client, model="bytedance/seedance-2-5", generate_audio=True
        )
        assert with_flag == {**without, "generate_audio": True}

    @pytest.mark.parametrize(
        "model",
        [
            "bytedance/seedance-2",
            "bytedance/seedance-2-fast",
            "bytedance/seedance-2-mini",
        ],
    )
    @pytest.mark.parametrize("value", [True, False])
    def test_raises_for_a_model_that_does_not_support_it(self, model, value):
        """The capability is enforced at the boundary, not left to the caller.

        A 2.0-family request carrying `generate_audio` is rejected by kie.ai, so
        we want a ValueError here — before the clip is cut and uploaded — rather
        than an opaque 4xx (or worse, a caller that forgot to check
        spec.supports_generate_audio and silently ships a broken payload).
        False is rejected as loudly as True: it is the *presence* of the key the
        2.0 models object to.
        """
        client = _make_client()
        with pytest.raises(ValueError, match="generate_audio is not supported"):
            client.create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=8,
                model=model,
                generate_audio=value,
            )

    def test_raises_for_an_unknown_model_id(self):
        """An unregistered id resolves to the default (2.0) spec, which lacks it.

        Same fallback the duration validation uses — the registry is the
        authority, so an id it has never heard of gets the conservative rules.
        """
        client = _make_client()
        with pytest.raises(ValueError, match="generate_audio is not supported"):
            client.create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=8,
                model="bytedance/seedance-99-turbo",
                generate_audio=True,
            )

    def test_the_rejected_set_is_read_from_the_registry(self):
        """Every model whose spec lacks the capability is rejected, and only those.

        Pinned as a set comparison so adding a model with
        supports_generate_audio=True cannot leave this guard behind.
        """
        from app import ai_models

        client = _make_client()
        rejected = set()
        for key, spec in ai_models.AI_MODELS.items():
            if spec.family != "seedance":
                continue  # Omni has its own method, with no audio switch at all.
            # Send each model its OWN legal duration, or a duration ValueError
            # would masquerade as a capability rejection (2.5 only takes -1).
            duration = ai_models.duration_for(key, 0.0, 8.0)
            try:
                with respx.mock:
                    self._mock_create()
                    client.create_task(
                        prompt="x",
                        reference_image_urls=[],
                        reference_video_urls=[],
                        duration=duration,
                        model=spec.kie_model_id,
                        generate_audio=True,
                    )
            except ValueError:
                rejected.add(key)
        assert rejected == {
            k for k, s in ai_models.AI_MODELS.items()
            if s.family == "seedance" and not s.supports_generate_audio
        }


# ---------------------------------------------------------------------------
# create_omni_task
# ---------------------------------------------------------------------------


class TestCreateOmniTask:
    @respx.mock
    def test_returns_task_id_and_sends_correct_body(self):
        """create_omni_task sends the gemini-omni-video payload and returns taskId."""
        create_url = f"{JOBS_BASE}/api/v1/jobs/createTask"
        respx.post(create_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "success",
                    "data": {"taskId": "task_gemini_1", "recordId": "rec"},
                },
            )
        )

        client = _make_client()
        task_id = client.create_omni_task(
            prompt="replace the person",
            image_urls=["https://img.test/face.jpg"],
            video_url="https://vid.test/seg.mp4",
            video_start=0,
            video_end=8.0,
            resolution="1080p",
            aspect_ratio="9:16",
            duration=8,
        )
        assert task_id == "task_gemini_1"

        req = respx.calls.last.request
        body = json.loads(req.content)
        assert body["model"] == "gemini-omni-video"
        inp = body["input"]
        assert inp["prompt"] == "replace the person"
        assert inp["image_urls"] == ["https://img.test/face.jpg"]
        # video trim sent as ints; duration sent as a STRING enum.
        assert inp["video_list"] == [
            {"url": "https://vid.test/seg.mp4", "start": 0, "ends": 8}
        ]
        assert inp["resolution"] == "1080p"
        assert inp["aspect_ratio"] == "9:16"
        assert inp["duration"] == "8"
        assert "seed" not in inp  # omitted when not provided

    @respx.mock
    def test_null_data_200_raises_with_msg(self):
        """A 200 response with data=null surfaces the API msg, not an AttributeError."""
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200,
                json={"code": 422, "msg": "invalid duration", "data": None},
            )
        )
        client = _make_client()
        with pytest.raises(KieTaskError, match="invalid duration"):
            client.create_omni_task(
                prompt="x",
                image_urls=[],
                video_url="https://vid.test/seg.mp4",
                video_start=0,
                video_end=4.0,
                duration=4,
            )

    @respx.mock
    def test_seed_included_when_provided(self):
        """create_omni_task includes seed in the payload when given."""
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200, json={"code": 200, "data": {"taskId": "t2"}}
            )
        )
        client = _make_client()
        client.create_omni_task(
            prompt="x",
            image_urls=[],
            video_url="https://vid.test/seg.mp4",
            video_start=0,
            video_end=4.0,
            duration=4,
            seed=42,
        )
        body = json.loads(respx.calls.last.request.content)
        assert body["input"]["seed"] == 42

    def test_invalid_duration_raises(self):
        """create_omni_task raises ValueError for a duration outside the allowed set."""
        client = _make_client()
        with pytest.raises(ValueError, match="duration must be one of"):
            client.create_omni_task(
                prompt="x",
                image_urls=[],
                video_url="https://vid.test/seg.mp4",
                video_start=0,
                video_end=5.0,
                duration=5,  # not in (4, 6, 8, 10)
            )

    def test_duration_validation_follows_the_registry(self):
        """The allowed set is AI_MODELS["gemini-omni"].allowed_durations, not a copy.

        This method used to validate against a module-local ``_OMNI_DURATIONS``
        tuple — a duplicate of the registry entry that nothing kept in sync while
        create_task had already been converted. Driving both off the spec is what
        makes the two impossible to drift apart.
        """
        from app import ai_models

        allowed = ai_models.AI_MODELS["gemini-omni"].allowed_durations
        assert allowed  # a fixed-set model, by definition
        client = _make_client()

        for duration in allowed:
            with respx.mock:
                respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
                    return_value=httpx.Response(
                        200,
                        json={"code": 200, "msg": "ok", "data": {"taskId": "t"}},
                    )
                )
                client.create_omni_task(
                    prompt="x",
                    image_urls=[],
                    video_url="https://vid.test/seg.mp4",
                    video_start=0,
                    video_end=float(duration),
                    duration=duration,
                )

        for duration in (1, 3, 5, 7, 9, 11, 30):
            assert duration not in allowed
            with pytest.raises(ValueError, match="duration must be one of"):
                client.create_omni_task(
                    prompt="x",
                    image_urls=[],
                    video_url="https://vid.test/seg.mp4",
                    video_start=0,
                    video_end=float(duration),
                    duration=duration,
                )

    @respx.mock
    def test_raises_on_http_error(self):
        """create_omni_task raises KieTaskError on a 4xx response."""
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(403, json={"error": "Forbidden"})
        )
        client = _make_client()
        with pytest.raises(KieTaskError, match="403"):
            client.create_omni_task(
                prompt="x",
                image_urls=[],
                video_url="https://vid.test/seg.mp4",
                video_start=0,
                video_end=4.0,
                duration=4,
            )


# ---------------------------------------------------------------------------
# create_image_task (GPT Image image-to-image)
# ---------------------------------------------------------------------------


class TestCreateImageTask:
    @respx.mock
    def test_returns_task_id_and_sends_correct_body(self):
        """create_image_task sends the configured model's payload and returns taskId."""
        from app.config import settings

        create_url = f"{JOBS_BASE}/api/v1/jobs/createTask"
        respx.post(create_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "success",
                    "data": {"taskId": "task_img_1", "recordId": "rec"},
                },
            )
        )

        client = _make_client()
        task_id = client.create_image_task(
            prompt="make the face look like the reference photo",
            input_urls=["https://img.test/face.jpg"],
        )
        assert task_id == "task_img_1"

        req = respx.calls.last.request
        body = json.loads(req.content)
        assert body["model"] == settings.KIE_IMAGE_MODEL
        inp = body["input"]
        assert inp["prompt"] == "make the face look like the reference photo"
        assert inp["input_urls"] == ["https://img.test/face.jpg"]
        # aspect_ratio stays a literal default; resolution comes from settings.
        assert inp["aspect_ratio"] == "auto"
        assert inp["resolution"] == settings.KIE_IMAGE_RESOLUTION

    @respx.mock
    def test_forwards_overridden_aspect_ratio_and_resolution(self):
        """create_image_task passes through aspect_ratio/resolution kwargs."""
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200, json={"code": 200, "data": {"taskId": "t_img_2"}}
            )
        )
        client = _make_client()
        client.create_image_task(
            prompt="x",
            input_urls=["https://img.test/a.png", "https://img.test/b.png"],
            aspect_ratio="9:16",
            resolution="2K",
        )
        inp = json.loads(respx.calls.last.request.content)["input"]
        assert inp["aspect_ratio"] == "9:16"
        assert inp["resolution"] == "2K"
        assert len(inp["input_urls"]) == 2

    @respx.mock
    def test_null_data_200_raises_with_msg(self):
        """A 200 response with data=null surfaces the API msg."""
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200,
                json={"code": 422, "msg": "invalid input", "data": None},
            )
        )
        client = _make_client()
        with pytest.raises(KieTaskError, match="invalid input"):
            client.create_image_task(prompt="x", input_urls=[])

    @respx.mock
    def test_raises_on_http_error(self):
        """create_image_task raises KieTaskError on a 4xx response."""
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(403, json={"error": "Forbidden"})
        )
        client = _make_client()
        with pytest.raises(KieTaskError, match="403"):
            client.create_image_task(prompt="x", input_urls=[])

    def test_signature_defers_model_and_resolution_to_settings(self):
        """`model` and `resolution` default to None so settings win at CALL time.

        A signature default (``resolution: str = "1K"``) would be evaluated once
        at import and freeze whatever the env said then — the env knob would
        then be unsettable from a test, and a worker started before an .env
        edit would keep the old value. ``aspect_ratio`` stays a literal because
        every caller sends ``auto`` and all three model variants accept it.
        """
        import inspect

        from app.kie_client import KieClient

        sig = inspect.signature(KieClient.create_image_task)
        assert sig.parameters["aspect_ratio"].default == "auto"
        assert sig.parameters["resolution"].default is None
        assert sig.parameters["model"].default is None

    def test_shipped_default_is_gpt_image_2_5_flare(self):
        """The default model is flare — changing it must be a deliberate edit.

        gpt-image-2, 2.5-flare and 2.5-sunburst all cost 6 credits at 1K
        (measured live 2026-09-12), so nothing but this constant decides which
        one every run pays for.
        """
        from app.config import settings

        assert settings.KIE_IMAGE_MODEL == "gpt-image-2-5-flare-image-to-image"
        assert settings.KIE_IMAGE_RESOLUTION == "1K"

    @respx.mock
    def test_reads_settings_at_call_time(self, monkeypatch):
        """Changing the setting changes the NEXT task — no re-import needed."""
        from app.config import settings

        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200, json={"code": 200, "data": {"taskId": "t_img_cfg"}}
            )
        )
        monkeypatch.setattr(
            settings, "KIE_IMAGE_MODEL", "gpt-image-2-5-sunburst-image-to-image"
        )
        monkeypatch.setattr(settings, "KIE_IMAGE_RESOLUTION", "2K")

        client = _make_client()
        client.create_image_task(prompt="x", input_urls=["https://img.test/a.png"])

        body = json.loads(respx.calls.last.request.content)
        assert body["model"] == "gpt-image-2-5-sunburst-image-to-image"
        assert body["input"]["resolution"] == "2K"

    @respx.mock
    def test_per_call_model_override_beats_settings(self, monkeypatch):
        """An explicit `model=` wins — the hook for a per-task escalation."""
        from app.config import settings

        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200, json={"code": 200, "data": {"taskId": "t_img_ovr"}}
            )
        )
        monkeypatch.setattr(
            settings, "KIE_IMAGE_MODEL", "gpt-image-2-5-flare-image-to-image"
        )

        client = _make_client()
        client.create_image_task(
            prompt="x",
            input_urls=["https://img.test/a.png"],
            model="gpt-image-2-image-to-image",
        )
        assert (
            json.loads(respx.calls.last.request.content)["model"]
            == "gpt-image-2-image-to-image"
        )


# ---------------------------------------------------------------------------
# poll_task
# ---------------------------------------------------------------------------


class TestPollTask:
    def _record_response(self, state: str, **extra) -> dict:
        return {"code": 200, "data": {"state": state, **extra}}

    def _success_response(self) -> dict:
        result_json = json.dumps(
            {"resultUrls": ["https://result.test/output.mp4"]}
        )
        return {
            "code": 200,
            "data": {"state": "success", "resultJson": result_json},
        }

    @respx.mock
    def test_waiting_generating_success_sequence(self):
        """poll_task progresses through waiting→generating→success and returns URL."""
        poll_url = f"{JOBS_BASE}/api/v1/jobs/recordInfo"
        sleep_log: list[float] = []

        responses = [
            httpx.Response(200, json={"code": 200, "data": {"state": "waiting"}}),
            httpx.Response(200, json={"code": 200, "data": {"state": "generating"}}),
            httpx.Response(200, json=self._success_response()),
        ]
        respx.get(poll_url).mock(side_effect=responses)

        client = _make_client(sleep_calls=sleep_log)
        result_url = client.poll_task("task-abc", interval_sec=5.0, timeout_sec=300.0)

        assert result_url == "https://result.test/output.mp4"
        # sleep called twice (after waiting and after generating)
        assert sleep_log == [5.0, 5.0]

    @respx.mock
    def test_fail_state_raises_kie_task_failed(self):
        """poll_task raises KieTaskFailed with the failMsg when state=fail."""
        poll_url = f"{JOBS_BASE}/api/v1/jobs/recordInfo"
        respx.get(poll_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": {
                        "state": "fail",
                        "failMsg": "insufficient credits",
                        "failCode": 4001,
                    },
                },
            )
        )

        client = _make_client()
        with pytest.raises(KieTaskFailed, match="insufficient credits") as exc_info:
            client.poll_task("task-fail")

        assert exc_info.value.fail_msg == "insufficient credits"

    @respx.mock
    def test_timeout_raises_kie_task_error(self):
        """poll_task raises KieTaskError after timeout_sec elapsed."""
        poll_url = f"{JOBS_BASE}/api/v1/jobs/recordInfo"
        # Always return 'queuing' so we never finish.
        respx.get(poll_url).mock(
            return_value=httpx.Response(
                200, json={"code": 200, "data": {"state": "queuing"}}
            )
        )

        # Use a counter-based monotonic: each call increments by a large step
        # so the deadline is exceeded after the first poll.  We use a mutable
        # list as a counter so the closure can mutate it.
        _counter = [0.0]

        def _fake_monotonic() -> float:
            val = _counter[0]
            _counter[0] += 500.0  # each call advances "time" by 500 s
            return val

        import app.kie_client as kie_mod

        client = _make_client()
        with patch.object(kie_mod.time, "monotonic", side_effect=_fake_monotonic):
            with pytest.raises(KieTaskError, match="timed out"):
                client.poll_task("task-hang", interval_sec=0.0, timeout_sec=1.0)


# ---------------------------------------------------------------------------
# download_result
# ---------------------------------------------------------------------------


class TestDownloadResult:
    @respx.mock
    def test_streams_bytes_to_file(self, tmp_path):
        """download_result writes streamed bytes to the destination path."""
        dst = tmp_path / "output.mp4"
        video_bytes = b"\x00\x01\x02fake-video-content"

        respx.get("https://result.test/output.mp4").mock(
            return_value=httpx.Response(200, content=video_bytes)
        )

        client = _make_client()
        client.download_result("https://result.test/output.mp4", str(dst))

        assert dst.exists()
        assert dst.read_bytes() == video_bytes

    @respx.mock
    def test_raises_on_non_200(self, tmp_path):
        """download_result raises KieError on a non-200 response."""
        dst = tmp_path / "output.mp4"
        respx.get("https://result.test/fail.mp4").mock(
            return_value=httpx.Response(404)
        )

        from app.kie_client import KieError

        client = _make_client()
        with pytest.raises(KieError, match="404"):
            client.download_result("https://result.test/fail.mp4", str(dst))

    @respx.mock
    def test_retries_a_truncated_body_and_succeeds(self, tmp_path):
        """A body cut short mid-stream is retried, not accepted.

        Regression for run ce9b5bfd: the result CDN closed the connection after
        1805952 of 2249658 bytes, which failed the whole run over an image that
        was sitting on kie.ai intact.
        """
        dst = tmp_path / "ref_0001.png"
        good = b"complete-image-bytes"
        calls = {"n": 0}

        def _responder(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body",
                    request=request,
                )
            return httpx.Response(200, content=good)

        respx.get("https://result.test/img.png").mock(side_effect=_responder)

        client = _make_client()
        client.download_result("https://result.test/img.png", str(dst))

        assert calls["n"] == 2
        assert dst.read_bytes() == good

    @respx.mock
    def test_short_body_against_content_length_is_rejected(self, tmp_path):
        """A body shorter than the advertised length never reaches dst_path."""
        dst = tmp_path / "ref_0001.png"
        respx.get("https://result.test/short.png").mock(
            return_value=httpx.Response(
                200, content=b"tiny", headers={"content-length": "999999"},
            )
        )

        client = _make_client()
        with pytest.raises(httpx.RemoteProtocolError):
            client.download_result("https://result.test/short.png", str(dst))

        # No truncated file left behind for a later phase to mistake for a
        # finished download, and no stray .part either.
        assert not dst.exists()
        assert not dst.with_name(dst.name + ".part").exists()

    @respx.mock
    def test_gone_result_url_raises_kie_result_gone(self, tmp_path):
        """A 4xx is distinguishable from a transient drop.

        The two want opposite recoveries — keep the task id and re-download, or
        drop it and generate again.
        """
        from app.kie_client import KieResultGone

        dst = tmp_path / "ref_0001.png"
        respx.get("https://result.test/gone.png").mock(
            return_value=httpx.Response(404)
        )

        client = _make_client()
        with pytest.raises(KieResultGone):
            client.download_result("https://result.test/gone.png", str(dst))
        assert not dst.exists()


# ---------------------------------------------------------------------------
# get_credit_balance
# ---------------------------------------------------------------------------


class TestGetCreditBalance:
    @respx.mock
    def test_returns_balance(self):
        client = _make_client()
        url = f"{JOBS_BASE}/api/v1/chat/credit"
        respx.get(url).mock(
            return_value=httpx.Response(
                200, json={"code": 200, "msg": "success", "data": 4200}
            )
        )
        assert client.get_credit_balance() == 4200.0

    @respx.mock
    def test_raises_on_missing_data(self):
        client = _make_client()
        url = f"{JOBS_BASE}/api/v1/chat/credit"
        respx.get(url).mock(
            return_value=httpx.Response(200, json={"code": 402, "msg": "no data"})
        )
        with pytest.raises(Exception):
            client.get_credit_balance()

    @respx.mock
    def test_raises_on_http_error(self):
        client = _make_client()
        url = f"{JOBS_BASE}/api/v1/chat/credit"
        respx.get(url).mock(return_value=httpx.Response(401, json={"msg": "unauth"}))
        with pytest.raises(Exception):
            client.get_credit_balance()


# ---------------------------------------------------------------------------
# upload_file retry seek (Bug 1)
# ---------------------------------------------------------------------------


class TestUploadRetrySeek:
    @respx.mock
    def test_retry_resends_full_body_after_transient_500(self, tmp_path):
        """On a transient 500 the retry must seek(0) and re-send the full body."""
        src = tmp_path / "clip.mp4"
        file_content = b"real-video-bytes"
        src.write_bytes(file_content)

        upload_url = f"{UPLOAD_BASE}/api/file-stream-upload"
        success_body = {
            "success": True,
            "code": 200,
            "data": {"downloadUrl": "https://tempfile.test/clip.mp4"},
        }
        # First call → 500 (transient), second call → 200 success.
        respx.post(upload_url).mock(
            side_effect=[
                httpx.Response(500, text="internal error"),
                httpx.Response(200, json=success_body),
            ]
        )

        client = _make_client()
        url = client.upload_file(str(src), upload_path="charswap")
        assert url == "https://tempfile.test/clip.mp4"
        assert respx.calls.call_count == 2

        # Both requests must contain the full file content.
        for call in respx.calls:
            body = call.request.content
            assert file_content in body, (
                f"Retry attempt sent truncated body (len={len(body)}); "
                "seek(0) not being called before each attempt"
            )

    @respx.mock
    def test_first_attempt_reads_full_body(self, tmp_path):
        """Even on the first (non-retry) attempt the full body is uploaded."""
        src = tmp_path / "data.bin"
        file_content = b"\xde\xad\xbe\xef" * 256
        src.write_bytes(file_content)

        upload_url = f"{UPLOAD_BASE}/api/file-stream-upload"
        respx.post(upload_url).mock(
            return_value=httpx.Response(
                200,
                json={"success": True, "code": 200, "data": {"downloadUrl": "https://tempfile.test/data.bin"}},
            )
        )

        client = _make_client()
        client.upload_file(str(src))

        body = respx.calls.last.request.content
        assert file_content in body


# ---------------------------------------------------------------------------
# KieClient context manager and close (Bug 2)
# ---------------------------------------------------------------------------


class TestKieClientLifecycle:
    def test_context_manager_calls_close(self):
        """KieClient can be used as a context manager; __exit__ calls close()."""
        closed = []

        with _make_client() as client:
            # Patch close() so we can detect it was called without touching the
            # real httpx.Client (no network needed).
            original_close = client.close
            client.close = lambda: closed.append(True)

        assert closed == [True], "close() was not called on context-manager exit"

    def test_close_method_exists(self):
        """KieClient exposes a close() method."""
        client = _make_client()
        assert callable(getattr(client, "close", None))
        # Actually close it to exercise the path (httpx.Client.close is idempotent).
        client.close()


# ---------------------------------------------------------------------------
# get_shared_client (Bug 2 — module-level singleton)
# ---------------------------------------------------------------------------


class TestGetSharedClient:
    def test_returns_kie_client_instance(self):
        """get_shared_client() returns a KieClient."""
        import app.kie_client as kie_mod

        # Reset module-level singleton so this test is independent.
        orig = kie_mod._shared_client
        kie_mod._shared_client = None
        try:
            client = get_shared_client()
            assert isinstance(client, KieClient)
        finally:
            kie_mod._shared_client = orig

    def test_returns_same_instance_on_repeated_calls(self):
        """get_shared_client() is idempotent — same object returned each time."""
        import app.kie_client as kie_mod

        orig = kie_mod._shared_client
        kie_mod._shared_client = None
        try:
            c1 = get_shared_client()
            c2 = get_shared_client()
            assert c1 is c2
        finally:
            kie_mod._shared_client = orig


# ---------------------------------------------------------------------------
# create_task — the Seedance 2.5 `reference_audio_urls` voice reference
# ---------------------------------------------------------------------------


class TestCreateTaskReferenceAudio:
    """Voice references reach the wire only for models whose spec takes them.

    Same contract as `generate_audio` above and for the same reason: a model
    without the field rejects the whole request, so "no voice" must mean the key
    is absent from the JSON body, not present-and-empty.
    """

    @staticmethod
    def _mock_create():
        respx.post(f"{JOBS_BASE}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(
                200,
                json={"code": 200, "msg": "success", "data": {"taskId": "t-voice"}},
            )
        )

    def _create(self, client, **kw) -> dict:
        from app import ai_models
        model_id = kw.get("model", "bytedance/seedance-2")
        key = next(
            (k for k, sp in ai_models.AI_MODELS.items()
             if sp.kie_model_id == model_id),
            "seedance",
        )
        kw.setdefault("duration", ai_models.duration_for(key, 0.0, 8.0))
        client.create_task(
            prompt="swap the character",
            reference_image_urls=[],
            reference_video_urls=["https://vid.test/seg.mp4"],
            **kw,
        )
        return json.loads(respx.calls.last.request.content)["input"]

    @respx.mock
    def test_included_in_input_when_passed(self):
        self._mock_create()
        inp = self._create(
            _make_client(),
            model="bytedance/seedance-2-5",
            reference_audio_urls=["https://aud.test/voice.mp3"],
        )
        assert inp["reference_audio_urls"] == ["https://aud.test/voice.mp3"]

    @respx.mock
    def test_key_absent_when_not_passed(self):
        """Absent, not empty — this is what keeps 2.0 payloads unchanged."""
        self._mock_create()
        inp = self._create(_make_client(), model="bytedance/seedance-2")
        assert "reference_audio_urls" not in inp

    @respx.mock
    def test_empty_list_is_omitted(self):
        """[] means "no voice", which must not put an empty key on the wire.

        The pipeline always passes a list (possibly empty) rather than None, so
        this is the common path, not an edge case.
        """
        self._mock_create()
        inp = self._create(
            _make_client(), model="bytedance/seedance-2-5", reference_audio_urls=[]
        )
        assert "reference_audio_urls" not in inp

    @respx.mock
    def test_adding_a_voice_changes_nothing_else_in_the_payload(self):
        self._mock_create()
        client = _make_client()
        without = self._create(client, model="bytedance/seedance-2-5")
        with_voice = self._create(
            client,
            model="bytedance/seedance-2-5",
            reference_audio_urls=["https://aud.test/v.mp3"],
        )
        assert with_voice == {
            **without,
            "reference_audio_urls": ["https://aud.test/v.mp3"],
        }

    def test_default_is_none(self):
        import inspect

        sig = inspect.signature(KieClient.create_task)
        assert sig.parameters["reference_audio_urls"].default is None

    @respx.mock
    def test_rejected_for_a_model_without_the_field(self):
        """A 2.0-family request carrying a voice is rejected by kie.ai.

        Caught here as a ValueError instead of as an opaque 4xx *after* the clip
        has been cut and uploaded — the same reasoning as the generate_audio
        guard above.
        """
        self._mock_create()
        with pytest.raises(ValueError, match="reference_audio_urls is not supported"):
            _make_client().create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=8,
                model="bytedance/seedance-2",
                reference_audio_urls=["https://aud.test/v.mp3"],
            )

    @respx.mock
    def test_rejected_above_the_models_count_cap(self):
        """Truncating silently would hand back a generation nobody asked for."""
        from app import ai_models

        cap = ai_models.AI_MODELS["seedance-2-5"].max_reference_audios
        self._mock_create()
        with pytest.raises(ValueError, match="at most"):
            _make_client().create_task(
                prompt="x",
                reference_image_urls=[],
                reference_video_urls=[],
                duration=ai_models.duration_for("seedance-2-5", 0.0, 8.0),
                model="bytedance/seedance-2-5",
                reference_audio_urls=[
                    f"https://aud.test/v{i}.mp3" for i in range(cap + 1)
                ],
            )

    def test_the_rejected_set_is_read_from_the_registry(self):
        """Every model whose spec lacks the field is rejected, and only those.

        Pinned as a set comparison so adding a model with
        max_reference_audios > 0 cannot leave this guard behind.
        """
        from app import ai_models

        client = _make_client()
        rejected = set()
        for key, spec in ai_models.AI_MODELS.items():
            if spec.family != "seedance":
                continue  # Omni has its own method, with no voice reference.
            duration = ai_models.duration_for(key, 0.0, 8.0)
            try:
                with respx.mock:
                    self._mock_create()
                    client.create_task(
                        prompt="x",
                        reference_image_urls=[],
                        reference_video_urls=[],
                        duration=duration,
                        model=spec.kie_model_id,
                        reference_audio_urls=["https://aud.test/v.mp3"],
                    )
            except ValueError:
                rejected.add(key)
        assert rejected == {
            k for k, s in ai_models.AI_MODELS.items()
            if s.family == "seedance" and not s.max_reference_audios
        }


# ---------------------------------------------------------------------------
# LLM routes: kie.ai's HTTP-200 error envelopes
# ---------------------------------------------------------------------------


class TestLlmEnvelopeRetry:
    """kie.ai answers "being maintained" as HTTP 200 + {"code":500} (seen live on
    gemini-2.5-pro 2026-09-25). That must be retried like an HTTP 5xx, then
    surface as KieModelUnavailableError so a caller can switch models."""

    CHAT = f"{JOBS_BASE}/gemini-2.5-pro/v1/chat/completions"
    RESPONSES = f"{JOBS_BASE}/codex/v1/responses"
    OK = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
    MAINT = {"code": 500, "msg": "The server is currently being maintained"}

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(KieClient._chat_with_retry.retry, "sleep", lambda _s: None)

    def _chat(self, client):
        return client.chat_completion(
            model="gemini-2.5-pro", messages=[{"role": "user", "content": "hi"}]
        )

    @respx.mock
    def test_envelope_500_is_retried_then_succeeds(self):
        route = respx.post(self.CHAT).mock(
            side_effect=[
                httpx.Response(200, json=self.MAINT),
                httpx.Response(200, json=self.OK),
            ]
        )
        assert self._chat(_make_client()) == "OK"
        assert route.call_count == 2

    @respx.mock
    def test_envelope_500_exhausted_is_unavailable(self):
        route = respx.post(self.CHAT).mock(
            return_value=httpx.Response(200, json=self.MAINT)
        )
        with pytest.raises(KieModelUnavailableError, match=r"gemini-2\.5-pro.*code=500"):
            self._chat(_make_client())
        assert route.call_count == 4

    @respx.mock
    def test_http_5xx_exhausted_is_unavailable(self):
        respx.post(self.CHAT).mock(return_value=httpx.Response(502, text="bad gw"))
        with pytest.raises(KieModelUnavailableError, match="HTTP 502"):
            self._chat(_make_client())

    @respx.mock
    def test_envelope_422_is_unavailable_without_retry(self):
        route = respx.post(self.CHAT).mock(
            return_value=httpx.Response(
                200, json={"code": 422, "msg": "The channel is not supported"}
            )
        )
        with pytest.raises(KieModelUnavailableError, match="code=422"):
            self._chat(_make_client())
        assert route.call_count == 1

    @respx.mock
    def test_envelope_404_invalid_model_is_unavailable_without_retry(self):
        # kie.ai's answer for a retired model (gemini-2.5-pro, 2026-10-13).
        route = respx.post(self.CHAT).mock(
            return_value=httpx.Response(
                200, json={"code": 404, "msg": "Invalid model name: HTTP 404"}
            )
        )
        with pytest.raises(KieModelUnavailableError, match="code=404"):
            self._chat(_make_client())
        assert route.call_count == 1

    @respx.mock
    def test_envelope_401_is_plain_chat_error(self):
        route = respx.post(self.CHAT).mock(
            return_value=httpx.Response(200, json={"code": 401, "msg": "Unauthorized"})
        )
        with pytest.raises(KieChatError, match="code=401") as exc_info:
            self._chat(_make_client())
        assert not isinstance(exc_info.value, KieModelUnavailableError)
        assert route.call_count == 1

    @respx.mock
    def test_responses_route_shares_the_retry(self):
        route = respx.post(self.RESPONSES).mock(
            side_effect=[
                httpx.Response(200, json=self.MAINT),
                httpx.Response(
                    200,
                    json={"output": [{"type": "message", "content": [
                        {"type": "output_text", "text": "hola"}
                    ]}]},
                ),
            ]
        )
        out = _make_client().create_response(model="gpt-5-6-luna", input_text="hi")
        assert out == "hola"
        assert route.call_count == 2


class TestCreateResponseImages:
    RESPONSES = f"{JOBS_BASE}/codex/v1/responses"

    @respx.mock
    def test_images_follow_the_text_as_captioned_data_urls(self):
        route = respx.post(self.RESPONSES).mock(
            return_value=httpx.Response(
                200,
                json={"output": [{"type": "message", "content": [
                    {"type": "output_text", "text": "ok"}
                ]}]},
            )
        )
        out = _make_client().create_response(
            model="gpt-6-1-sol", input_text="hi", images=[("Frame at 1.0s:", b"abc")]
        )
        assert out == "ok"
        content = json.loads(route.calls[0].request.content)["input"][0]["content"]
        assert content == [
            {"type": "input_text", "text": "hi"},
            {"type": "input_text", "text": "Frame at 1.0s:"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,YWJj"},
        ]
