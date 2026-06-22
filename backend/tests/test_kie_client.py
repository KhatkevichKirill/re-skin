"""
Tests for KieClient — no real network calls, no real API key.

Uses respx to intercept httpx requests.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import httpx
import pytest
import respx

from app.kie_client import (
    KieClient,
    KieTaskError,
    KieTaskFailed,
    KieUploadError,
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
        assert inp["video_list"] == [
            {"url": "https://vid.test/seg.mp4", "start": 0, "ends": 8.0}
        ]
        assert inp["resolution"] == "1080p"
        assert inp["aspect_ratio"] == "9:16"
        assert inp["duration"] == 8
        assert "seed" not in inp  # omitted when not provided

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
