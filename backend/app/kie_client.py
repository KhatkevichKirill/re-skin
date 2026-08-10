"""
HTTP client for kie.ai / Seedance video generation API.

Handles:
  - File upload to the temp-file host
  - Task creation on the jobs API
  - Polling until success or failure
  - Downloading the result to a local path
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app import ai_models
from app.config import settings

logger = logging.getLogger(__name__)

# Default base URLs — overridable in constructor so tests can point at mocks.
_UPLOAD_BASE = "https://kieai.redpandaai.co"
_JOBS_BASE = "https://api.kie.ai"

# Gemini Omni Video accepts only a fixed set of output durations (seconds).
_OMNI_DURATIONS = (4, 6, 8, 10)

# Reverse index kie model id -> registry key, so create_task can validate its
# `duration` / `generate_audio` arguments against the right AIModelSpec. Callers
# pass the kie id (that is what the API wants); the capability table is keyed by
# the run_model_enum label.
_KEY_BY_KIE_ID: dict[str, str] = {
    spec.kie_model_id: key for key, spec in ai_models.AI_MODELS.items()
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class KieError(Exception):
    """Base exception for all kie.ai client errors."""


class KieUploadError(KieError):
    """Raised when a file upload fails."""


class KieTaskError(KieError):
    """Raised when task creation, polling, or timeout occurs."""


class KieTaskFailed(KieTaskError):
    """Raised when the Seedance task itself reports a failure state."""

    def __init__(self, fail_msg: str) -> None:
        self.fail_msg = fail_msg
        super().__init__(f"Task failed: {fail_msg}")


# ---------------------------------------------------------------------------
# createTask response helpers
# ---------------------------------------------------------------------------


def _extract_task_id(data) -> str | None:
    """Safely pull ``data.data.taskId`` from a createTask response.

    The jobs API can return HTTP 200 with ``"data": null`` and a non-200 body
    ``code``/``msg`` when it rejects a task; guard against that null so we raise
    a useful error instead of an AttributeError.
    """
    if not isinstance(data, dict):
        return None
    inner = data.get("data")
    if not isinstance(inner, dict):
        return None
    return inner.get("taskId")


def _no_task_id_msg(data) -> str:
    """Build an informative error message including the API's code/msg."""
    code = data.get("code") if isinstance(data, dict) else None
    msg = data.get("msg") if isinstance(data, dict) else None
    return f"createTask returned no taskId (code={code}, msg={msg!r}): {data}"


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _is_transient(exc: BaseException) -> bool:
    """Return True for network errors or 5xx HTTP status errors."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _transient_retry(func):
    """Decorator: retry up to 4 attempts on transient errors with exponential backoff."""
    return retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        reraise=True,
    )(func)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class KieClient:
    """
    Synchronous HTTP client for kie.ai / Seedance.

    Args:
        api_key: Bearer token for both the upload and jobs APIs.
                 Defaults to ``settings.KIE_API_KEY``.
        upload_base: Override the upload host (useful in tests).
        jobs_base: Override the jobs API host (useful in tests).
        sleep_fn: Callable used for sleeping during ``poll_task``; defaults to
                  ``time.sleep``.  Inject a no-op in tests to avoid real waits.
    """

    def __init__(
        self,
        api_key: str | None = None,
        upload_base: str = _UPLOAD_BASE,
        jobs_base: str = _JOBS_BASE,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._api_key = api_key or settings.KIE_API_KEY
        if not self._api_key:
            raise KieError("KIE_API_KEY is not set")
        self._upload_base = upload_base.rstrip("/")
        self._jobs_base = jobs_base.rstrip("/")
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self._client = httpx.Client(timeout=60.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_file(
        self,
        local_path: str,
        upload_path: str = "charswap",
        file_name: str | None = None,
    ) -> str:
        """
        Upload a local file to the kie.ai temp-file host.

        Args:
            local_path: Path to the local file.
            upload_path: Remote directory hint (e.g. ``"charswap/segments"``).
            file_name: Override the file name; defaults to the basename of *local_path*.

        Returns:
            The public ``downloadUrl`` of the uploaded file.

        Raises:
            KieUploadError: On HTTP errors or a missing URL in the response.
        """
        path = Path(local_path)
        fname = file_name or path.name
        url = f"{self._upload_base}/api/file-stream-upload"
        logger.info("Uploading %s to kie.ai (%s/%s)", path, upload_path, fname)

        try:
            with path.open("rb") as fh:
                data = self._upload_with_retry(url, fh, upload_path, fname)
        except KieUploadError:
            raise
        except Exception as exc:
            raise KieUploadError(f"Upload failed: {exc}") from exc

        download_url: str | None = (
            data.get("data", {}).get("downloadUrl") if isinstance(data, dict) else None
        )
        if not download_url:
            raise KieUploadError(f"Upload response missing downloadUrl: {data}")

        logger.info("Upload complete: %s", download_url)
        return download_url

    def create_task(
        self,
        *,
        prompt: str,
        reference_image_urls: list[str],
        reference_video_urls: list[str],
        resolution: str = "480p",
        aspect_ratio: str = "9:16",
        duration: int,
        model: str = "bytedance/seedance-2",
        generate_audio: bool | None = None,
    ) -> str:
        """
        Create a Seedance task on the jobs API.

        Args:
            prompt: Text prompt for the generation.
            reference_image_urls: Up to 9 image URLs.
            reference_video_urls: Up to 3 video URLs (mp4/mov), each no longer
                than the model's per-clip ceiling — 15 s for Seedance 2.0,
                30 s for Seedance 2.5.
            resolution: ``480p``, ``720p``, or ``1080p``. Seedance 2.5 tops out
                at ``720p`` (see :mod:`app.ai_models`).
            aspect_ratio: One of ``1:1|4:3|16:9|9:16|21:9|adaptive``. Seedance
                2.5 accepts ONLY ``adaptive``.
            duration: Integer output seconds. The accepted value is per-model:
                [4, 15] for the Seedance 2.0 family, and ``-1`` ("match the
                input video") for Seedance 2.5, which rejects anything else.
            model: kie.ai model id, e.g. ``bytedance/seedance-2``,
                ``bytedance/seedance-2-fast``, ``bytedance/seedance-2-mini``, or
                ``bytedance/seedance-2-5``. Defaults to the base Seedance 2.0
                model.
            generate_audio: Seedance 2.5 only — ask the model to generate (True)
                or suppress (False) an audio track. Left as ``None`` the key is
                omitted from the payload entirely, which is required for the 2.0
                family: those models reject the field.

        Returns:
            The ``taskId`` string.

        Raises:
            ValueError: If *duration* is not a value this *model* accepts, or if
                *generate_audio* is passed for a model whose spec does not
                support it.
            KieTaskError: On HTTP or API errors.
        """
        # Validate against the registry entry behind this kie id rather than a
        # hardcoded range — 2.0 and 2.5 disagree on the ceiling, and 2.5 wants
        # the -1 sentinel.
        model_key = _KEY_BY_KIE_ID.get(model)
        ai_models.validate_duration(model_key, duration)

        # Enforce the audio-switch capability at the boundary rather than
        # trusting every caller to check spec.supports_generate_audio first.
        # The 2.0 family rejects the field outright, so a caller that passes it
        # has a bug we want to see as a ValueError here — not as an opaque kie.ai
        # 4xx after the clip has already been cut and uploaded. `None` means "not
        # passed" (the default) and is always fine.
        if generate_audio is not None:
            spec = ai_models.spec_for(model_key)
            if not spec.supports_generate_audio:
                supported = sorted(
                    k for k, s in ai_models.AI_MODELS.items()
                    if s.supports_generate_audio
                )
                raise ValueError(
                    f"generate_audio is not supported by model {model!r} "
                    f"({spec.key}); only {supported} accept it"
                )

        url = f"{self._jobs_base}/api/v1/jobs/createTask"
        payload = {
            "model": model,
            "input": {
                "prompt": prompt,
                "reference_image_urls": reference_image_urls,
                "reference_video_urls": reference_video_urls,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            },
        }
        # Only send the switch when the caller asked for it: a 2.0 request must
        # stay byte-identical to what it has always been.
        if generate_audio is not None:
            payload["input"]["generate_audio"] = bool(generate_audio)
        logger.info(
            "Creating Seedance task (model=%s, resolution=%s, duration=%ds, "
            "generate_audio=%s)",
            model, resolution, duration, generate_audio,
        )

        try:
            data = self._create_task_with_retry(url, payload)
        except KieTaskError:
            raise
        except Exception as exc:
            raise KieTaskError(f"create_task failed: {exc}") from exc

        task_id = _extract_task_id(data)
        if not task_id:
            raise KieTaskError(_no_task_id_msg(data))

        logger.info("Task created: taskId=%s", task_id)
        return task_id

    def create_omni_task(
        self,
        *,
        prompt: str,
        image_urls: list[str],
        video_url: str,
        video_start: float,
        video_end: float,
        resolution: str = "720p",
        aspect_ratio: str = "9:16",
        duration: int,
        seed: int | None = None,
    ) -> str:
        """
        Create a Gemini Omni Video task on the (shared) jobs API.

        Unlike Seedance, Gemini takes its reference clip via ``video_list`` (a
        single ``{url, start, ends}`` object, trim range <= 10s) and its
        reference images via ``image_urls``.

        Args:
            prompt: Text prompt for the generation.
            image_urls: Up to 7 reference image URLs.
            video_url: Public URL of the reference clip (<= 30s, <= 100MB).
            video_start: Trim start (seconds) within the clip.
            video_end: Trim end (seconds); ``video_end - video_start`` must be <= 10.
            resolution: ``720p``, ``1080p``, or ``4k``.
            aspect_ratio: ``16:9`` or ``9:16``.
            duration: Output length in seconds — one of 4, 6, 8, 10.
            seed: Optional reproducibility seed.

        Returns:
            The ``taskId`` string.

        Raises:
            ValueError: If *duration* is not one of the allowed values.
            KieTaskError: On HTTP or API errors.
        """
        if duration not in _OMNI_DURATIONS:
            raise ValueError(
                f"duration must be one of {_OMNI_DURATIONS}, got {duration}"
            )

        url = f"{self._jobs_base}/api/v1/jobs/createTask"
        payload = {
            "model": "gemini-omni-video",
            "input": {
                "prompt": prompt,
                "image_urls": image_urls,
                "video_list": [
                    {
                        "url": video_url,
                        "start": int(video_start),
                        "ends": int(round(video_end)),
                    }
                ],
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                # The API expects duration as a string enum ("4"/"6"/"8"/"10").
                "duration": str(duration),
            },
        }
        if seed is not None:
            payload["input"]["seed"] = seed
        logger.info(
            "Creating Gemini Omni task (resolution=%s, duration=%ds)",
            resolution, duration,
        )

        try:
            data = self._create_task_with_retry(url, payload)
        except KieTaskError:
            raise
        except Exception as exc:
            raise KieTaskError(f"create_omni_task failed: {exc}") from exc

        task_id = _extract_task_id(data)
        if not task_id:
            raise KieTaskError(_no_task_id_msg(data))

        logger.info("Omni task created: taskId=%s", task_id)
        return task_id

    def get_task(self, task_id: str) -> dict:
        """
        Fetch the current status of a task.

        Args:
            task_id: The task ID returned by :meth:`create_task`.

        Returns:
            The ``data`` dict from the API response.

        Raises:
            KieTaskError: On HTTP or API errors.
        """
        url = f"{self._jobs_base}/api/v1/jobs/recordInfo"
        try:
            return self._get_task_with_retry(url, task_id)
        except KieTaskError:
            raise
        except Exception as exc:
            raise KieTaskError(f"get_task failed: {exc}") from exc

    def poll_task(
        self,
        task_id: str,
        *,
        interval_sec: float = 10.0,
        timeout_sec: float = 900.0,
    ) -> str:
        """
        Poll a task until it reaches ``success`` or ``fail``.

        Args:
            task_id: The task ID to poll.
            interval_sec: Seconds to wait between polls.
            timeout_sec: Maximum total seconds before raising.

        Returns:
            The first element of ``resultUrls`` from the parsed ``resultJson``.

        Raises:
            KieTaskFailed: If the task transitions to ``fail``.
            KieTaskError: On timeout or unexpected response structure.
        """
        deadline = time.monotonic() + timeout_sec
        logger.info(
            "Polling task %s (interval=%.1fs, timeout=%.0fs)",
            task_id,
            interval_sec,
            timeout_sec,
        )

        while True:
            data = self.get_task(task_id)
            state = data.get("state", "")
            logger.debug("Task %s state=%s", task_id, state)

            if state == "success":
                result_json_str = data.get("resultJson", "")
                try:
                    result = json.loads(result_json_str)
                except json.JSONDecodeError as exc:
                    raise KieTaskError(
                        f"Could not parse resultJson: {result_json_str!r}"
                    ) from exc
                result_urls: list = result.get("resultUrls", [])
                if not result_urls:
                    raise KieTaskError(f"resultUrls is empty in resultJson: {result}")
                logger.info("Task %s succeeded: %s", task_id, result_urls[0])
                return result_urls[0]

            if state == "fail":
                fail_msg = data.get("failMsg") or data.get("failCode") or "unknown"
                logger.error("Task %s failed: %s", task_id, fail_msg)
                raise KieTaskFailed(str(fail_msg))

            # Still in-progress (waiting / queuing / generating)
            if time.monotonic() >= deadline:
                raise KieTaskError(
                    f"Task {task_id} timed out after {timeout_sec}s (last state={state!r})"
                )

            self._sleep(interval_sec)

    def download_result(self, url: str, dst_path: str) -> None:
        """
        Stream-download the result video to a local file.

        Args:
            url: The result URL (from :meth:`poll_task`).
            dst_path: Local destination path.

        Raises:
            KieError: On HTTP errors.
        """
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading result %s → %s", url, dst)

        with self._client.stream("GET", url) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise KieError(
                    f"Download failed with status {exc.response.status_code}: {url}"
                ) from exc
            with dst.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    fh.write(chunk)

        logger.info("Download complete: %s (%d bytes)", dst, dst.stat().st_size)

    # ------------------------------------------------------------------
    # Internal retry-wrapped helpers
    # ------------------------------------------------------------------

    @_transient_retry
    def _upload_with_retry(self, url: str, fh, upload_path: str, fname: str) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        resp = self._client.post(
            url,
            headers=headers,
            files={"file": (fname, fh, "application/octet-stream")},
            data={"uploadPath": upload_path, "fileName": fname},
        )
        if resp.status_code >= 500:
            resp.raise_for_status()
        if not resp.is_success:
            raise KieUploadError(
                f"Upload HTTP {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    @_transient_retry
    def _create_task_with_retry(self, url: str, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = self._client.post(url, headers=headers, json=payload)
        if resp.status_code >= 500:
            resp.raise_for_status()
        if not resp.is_success:
            raise KieTaskError(
                f"createTask HTTP {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    @_transient_retry
    def _get_task_with_retry(self, url: str, task_id: str) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        resp = self._client.get(url, headers=headers, params={"taskId": task_id})
        if resp.status_code >= 500:
            resp.raise_for_status()
        if not resp.is_success:
            raise KieTaskError(
                f"recordInfo HTTP {resp.status_code}: {resp.text[:300]}"
            )
        body = resp.json()
        data = body.get("data")
        if data is None:
            raise KieTaskError(f"recordInfo response missing 'data': {body}")
        return data
