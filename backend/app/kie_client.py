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
import threading
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


class KieChatError(KieError):
    """Raised by the two synchronous LLM endpoints (:meth:`KieClient.chat_completion`
    and :meth:`KieClient.create_response`) on HTTP errors, API error envelopes,
    or a response whose text content cannot be located."""


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


def _envelope_error(data) -> str | None:
    """Return kie.ai's error text when *data* is an error envelope, else None.

    kie.ai answers a rejected LLM request with **HTTP 200** and a body of
    ``{"code": 401, "msg": "Unauthorized ...", "data": null}`` — verified live
    against both ``/{model}/v1/chat/completions`` and ``/codex/v1/responses``
    with a bad key and with an unknown model. ``resp.is_success`` therefore
    proves nothing on these routes and every caller has to sniff the body.

    A successful chat/responses body carries no ``code`` key at all, so the
    presence of a non-2xx ``code`` is the reliable discriminator.
    """
    if not isinstance(data, dict):
        return f"unexpected response type {type(data).__name__}: {data!r}"
    code = data.get("code")
    if isinstance(code, int) and not (200 <= code < 300):
        return f"API error code={code}, msg={data.get('msg')!r}"
    return None


def _extract_chat_content(data) -> str | None:
    """Pull ``choices[0].message.content`` out of a chat/completions body.

    ``content`` is normally a string; the multi-part list form (the same shape
    requests use) is accepted too and flattened, since nothing in the spec
    forbids a model answering that way.
    """
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        return "".join(parts) if parts else None
    return None


def _extract_response_output_text(data) -> str | None:
    """Concatenate every ``output_text`` part in a Responses API body.

    ``output`` is a list whose items may be ``reasoning`` records as well as the
    assistant ``message``, and the position of the message is **not** fixed —
    a live ``gpt-5-6-luna`` call with ``effort=medium`` came back with the
    message at index 0 and no reasoning item at all, so indexing ``output[1]``
    (as the published example does) is wrong. Scan instead.
    """
    if not isinstance(data, dict):
        return None
    output = data.get("output")
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts) if parts else None


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

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        timeout_sec: float | None = None,
    ) -> str:
        """
        Call a kie.ai model on the OpenAI-compatible chat route and return its text.

        This is a **synchronous** endpoint — unlike :meth:`create_task` and
        friends there is no taskId and nothing to poll; the model's answer comes
        back in the same response. Used by :mod:`app.localisation` to transcribe
        a source clip (see docs/localisation.md §4.2).

        Two things about this route are unlike the jobs API:

        * The model is a **URL path segment**, not a body field:
          ``POST https://api.kie.ai/{model}/v1/chat/completions``. An unknown
          slug comes back as ``code=422 "The model is not supported"``.
        * Every media type — image, video, audio, PDF — travels in the
          ``image_url`` envelope::

              {"type": "image_url", "image_url": {"url": "<public url>"}}

          Produce that URL with :meth:`upload_file`.

        Args:
            model: kie.ai model slug, e.g. ``gemini-2.5-pro``. Goes in the path.
            messages: OpenAI-style message list. Content may be a plain string
                or a list of ``{"type": "text"|"image_url", ...}`` parts.
            response_format: Optional ``{"type": "json_schema", "json_schema":
                {...}}`` block. Sent when given, but treat it as a hint, not a
                guarantee: Gemini 2.5 Pro has been observed returning a
                ```` ```json ```` fenced object even with ``strict: true``, so
                callers must still parse tolerantly.
            timeout_sec: Per-request timeout. Defaults to the client's own 60s,
                which is usually too tight for a request carrying a video.

        Returns:
            ``choices[0].message.content`` as a string.

        Raises:
            KieChatError: On HTTP errors, on kie.ai's HTTP-200 error envelope,
                or when the response carries no text content.
        """
        url = f"{self._jobs_base}/{model}/v1/chat/completions"
        payload: dict = {"messages": messages, "stream": False}
        if response_format is not None:
            payload["response_format"] = response_format
        logger.info(
            "chat_completion (model=%s, messages=%d, json_schema=%s)",
            model, len(messages), response_format is not None,
        )

        try:
            data = self._chat_with_retry(url, payload, timeout_sec)
        except KieChatError:
            raise
        except Exception as exc:
            raise KieChatError(f"chat_completion failed: {exc}") from exc

        err = _envelope_error(data)
        if err:
            raise KieChatError(f"chat_completion {model}: {err}")

        content = _extract_chat_content(data)
        if content is None:
            raise KieChatError(
                f"chat_completion {model}: no message content in response: "
                f"{str(data)[:300]}"
            )
        logger.info(
            "chat_completion ok (model=%s, chars=%d, usage=%s)",
            model, len(content), data.get("usage"),
        )
        return content

    def create_response(
        self,
        *,
        model: str,
        input_text: str,
        reasoning_effort: str = "medium",
        timeout_sec: float | None = None,
    ) -> str:
        """
        Call a kie.ai model on the Responses route and return its text.

        Also synchronous (no taskId). Used by :mod:`app.localisation` to
        translate a transcript (see docs/localisation.md §4.2). The shape is
        deliberately *not* shared with :meth:`chat_completion` — this route is a
        different API, not a variant of the same one:

        * the model travels in the **body**, the path is the fixed
          ``/codex/v1/responses``;
        * content parts are ``input_text`` / ``input_image`` — there is no
          ``image_url`` envelope and **no** ``response_format``, so JSON output
          has to be asked for in the prompt and parsed tolerantly;
        * ``reasoning.effort`` defaults to ``low`` server-side, which is why
          this method defaults it to ``medium``;
        * the answer is an ``output`` **list** that may contain ``reasoning``
          items before the assistant ``message``.

        Args:
            model: kie.ai model id, e.g. ``gpt-5-6-luna``. Goes in the body.
            input_text: The single user turn, sent as one ``input_text`` part.
            reasoning_effort: ``low`` | ``medium`` | ``high`` | ``xhigh``.
            timeout_sec: Per-request timeout; defaults to the client's own 60s.

        Returns:
            The assistant message text (all ``output_text`` parts concatenated).

        Raises:
            KieChatError: On HTTP errors, on kie.ai's HTTP-200 error envelope,
                or when the response carries no ``output_text``.
        """
        url = f"{self._jobs_base}/codex/v1/responses"
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_text}],
                }
            ],
            "reasoning": {"effort": reasoning_effort},
            "stream": False,
        }
        logger.info(
            "create_response (model=%s, effort=%s, chars=%d)",
            model, reasoning_effort, len(input_text),
        )

        try:
            data = self._chat_with_retry(url, payload, timeout_sec)
        except KieChatError:
            raise
        except Exception as exc:
            raise KieChatError(f"create_response failed: {exc}") from exc

        err = _envelope_error(data)
        if err:
            raise KieChatError(f"create_response {model}: {err}")

        content = _extract_response_output_text(data)
        if not content:
            raise KieChatError(
                f"create_response {model}: no output_text in response "
                f"(status={data.get('status')!r}): {str(data.get('output'))[:300]}"
            )
        logger.info(
            "create_response ok (model=%s, chars=%d, usage=%s)",
            model, len(content), data.get("usage"),
        )
        return content

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
    def _chat_with_retry(
        self, url: str, payload: dict, timeout_sec: float | None
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        kwargs: dict = {}
        if timeout_sec is not None:
            kwargs["timeout"] = timeout_sec
        resp = self._client.post(url, headers=headers, json=payload, **kwargs)
        if resp.status_code >= 500:
            resp.raise_for_status()
        if not resp.is_success:
            raise KieChatError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise KieChatError(
                f"LLM response is not JSON: {resp.text[:300]}"
            ) from exc

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


# ---------------------------------------------------------------------------
# Module-level shared instance
# ---------------------------------------------------------------------------

_shared_client: KieClient | None = None
_shared_client_lock = threading.Lock()


def get_shared_client() -> KieClient:
    """Return a lazily-created, module-level KieClient (thread-safe).

    Used by callers that make a single request and have no client of their own to
    thread through (app/localisation.py). Everything on the run path is handed an
    explicit client so tests can inject a fake.
    """
    global _shared_client
    if _shared_client is None:
        with _shared_client_lock:
            if _shared_client is None:
                _shared_client = KieClient()
    return _shared_client
