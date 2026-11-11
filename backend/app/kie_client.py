"""
HTTP client for kie.ai / Seedance video generation API.

Handles:
  - File upload to the temp-file host
  - Task creation on the jobs API
  - Polling until success or failure
  - Downloading the result to a local path
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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

# Ceiling on a single upload to the kie.ai temp-file host.
#
# Observed 2026-08-10: a 136.5 MB body came back HTTP 500 with an empty payload
# in 0.78s — far too fast to have been transferred (a 22.4 MB upload the same
# day took 2.3s), so the host refuses an oversized upload on Content-Length
# before reading it. The number matches the <=100MB ceiling kie.ai documents for
# the video inputs of its jobs API (see create_omni_task).
#
# Two things go wrong without this check, and both are worse than the limit:
# a 500 is classed transient (_is_transient), so tenacity spends four attempts
# on a deterministic refusal; and the operator is told "server error" for what
# is really "your file is too big", which points at the wrong fix entirely.
# Set KIE_UPLOAD_MAX_SIZE_MB=0 to disable the pre-flight check.
_UPLOAD_MAX_BYTES = int(
    float(os.environ.get("KIE_UPLOAD_MAX_SIZE_MB", "100")) * 1_000_000
)

# Reverse lookup: kie.ai model id → our internal registry key.
#
# create_task's public contract is the kie.ai id ("bytedance/seedance-2"), not
# our Run.model key — callers pass what goes on the wire. The duration rules,
# however, live in app.ai_models keyed by the internal name, and they now differ
# per model (2.0 tops out at 15s, 2.5 at 30s). Mapping back here keeps the
# signature stable while making validation registry-driven. An unrecognised id
# resolves to None, and ai_models.spec_for(None) falls back to the default
# (Seedance 2.0) spec — i.e. the historical 4-15s bound.
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


class KieTaskRefused(KieTaskError):
    """A single-send createTask (``once=True``) was REFUSED: an HTTP 4xx or
    kie.ai's HTTP-200 error envelope (``code != 200``, no taskId) on the one and
    only POST. Nothing was created, so nothing was billed (AICP-351 §5.1:
    ``failed_unbilled``)."""


class KieTaskUnknown(KieTaskError):
    """A single-send createTask (``once=True``) whose outcome is UNKNOWN: a 5xx,
    a transport error or timeout, an unparseable body, or a 2xx without a
    taskId. The request may have been accepted and billed, so it must never be
    resent blind (AICP-351 §5.1: the attempt is ``parked``)."""


class KieTaskFailed(KieTaskError):
    """Raised when the Seedance task itself reports a failure state."""

    def __init__(self, fail_msg: str) -> None:
        self.fail_msg = fail_msg
        super().__init__(f"Task failed: {fail_msg}")


class KieResultGone(KieError):
    """Raised when a result URL is permanently unavailable (a 4xx from the CDN).

    Distinguished from a transient download failure because the two want
    opposite recoveries: a transient failure keeps the kie.ai task id so a later
    retry re-downloads the SAME generated image for free, while a gone URL means
    the only way forward is to drop the task id and generate again (see
    ``pipeline_v2.poll_preprocess_once``).
    """


class KieChatError(KieError):
    """Raised by the two synchronous LLM endpoints (:meth:`KieClient.chat_completion`
    and :meth:`KieClient.create_response`) on HTTP errors, API error envelopes,
    or a response whose text content cannot be located."""


class KieModelUnavailableError(KieChatError):
    """The LLM route is down or does not serve this model right now.

    Raised instead of a plain :class:`KieChatError` when a *different* model
    might still answer: a 5xx/429 that outlived the retries (kie.ai's
    ``{"code":500,"msg":"The server is currently being maintained..."}``
    arrives as HTTP 200, observed live on gemini-2.5-pro 2026-09-25 while
    gemini-3.1-pro answered on the same key), a 422 ``"The model is not
    supported"`` / ``"The channel is not supported"``, or a 404 ``"Invalid
    model name"`` (kie.ai retired gemini-2.5-pro this way by 2026-10-13). A 401 or an
    unparseable answer stays a plain :class:`KieChatError` — switching models
    fixes neither. Callers use it to decide whether to fall back.

    ``retryable`` marks the 5xx/429 envelope case that ``_chat_with_retry``
    retries before letting it escape.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Single-attempt outcomes (rebuild paid-call protocol, design §7.2)
# ---------------------------------------------------------------------------

# Outcome kinds of create_task_once / chat_completion_once. NOT exceptions: the
# caller records each one as a row state, and only "refused" proves that
# nothing was billed.
# Timeout of the one createTask POST of a ``once=True`` create (AICP-351);
# the same 60 s the client and the rebuild single-send protocol use.
CREATE_ONCE_TIMEOUT_SEC = 60.0

OUTCOME_ACCEPTED = "accepted"   # the provider took the request (task id / answer)
OUTCOME_REFUSED = "refused"     # a complete, definite refusal: nothing billed
OUTCOME_UNKNOWN = "unknown"     # the request may have been taken and billed


@dataclass(frozen=True)
class SendOutcome:
    """What ONE provider POST did. ``kind`` is one of the OUTCOME_* constants.

    ``task_id`` is set for an accepted createTask, ``content`` for an accepted
    chat call; ``body`` is the parsed JSON response when there was one (an
    accepted chat call's whole body is stored as the row's response);
    ``http_status`` is None when no complete response arrived. ``file_unavailable``
    marks Gemini's ``400 Failed to get the file information`` refusal, which
    the storyboard classifies as "unavailable for this input" and answers by
    moving to the next model (a NEW row).
    """

    kind: str
    task_id: Optional[str] = None
    content: Optional[str] = None
    body: Optional[dict] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    file_unavailable: bool = False

    @property
    def accepted(self) -> bool:
        return self.kind == OUTCOME_ACCEPTED

    @property
    def refused(self) -> bool:
        return self.kind == OUTCOME_REFUSED

    @property
    def unknown(self) -> bool:
        return self.kind == OUTCOME_UNKNOWN


# LLM single-send classification (AICP-351 §5.2). ``kind``:
#   accepted     -> ``content`` is the answer (billed)
#   unavailable  -> 5xx / 429 / a 5xx-or-429 envelope: this model is down right
#                   now; the next fallback model may answer; the call MAY have
#                   billed (keep the estimate)
#   unsupported  -> 404/422 (HTTP or envelope) "model not supported/invalid":
#                   refused before acceptance (settle 0), next fallback model
#   refused      -> any other 4xx / error envelope: refused before acceptance
#                   (settle 0), a definite failure (another model will not fix it)
#   unknown      -> no complete response, unparseable body, 2xx without content
LLM_ACCEPTED = "accepted"
LLM_UNAVAILABLE = "unavailable"
LLM_UNSUPPORTED = "unsupported"
LLM_REFUSED = "refused"
LLM_UNKNOWN = "unknown"


@dataclass(frozen=True)
class LlmSend:
    kind: str
    content: Optional[str] = None
    body: Optional[dict] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    file_unavailable: bool = False


def classify_llm_response(status_code: int, body_text: str, extract) -> LlmSend:
    """Classify ONE complete LLM response (chat or responses route); *extract*
    pulls the answer text out of a 2xx body. Pure."""
    try:
        data = json.loads(body_text) if body_text else None
    except ValueError:
        data = None
    body = data if isinstance(data, dict) else None
    snippet = (body_text or "")[:300]
    if status_code >= 500 or status_code == 429:
        return LlmSend(LLM_UNAVAILABLE, body=body, http_status=status_code,
                       error=f"LLM HTTP {status_code}: {snippet}")
    if status_code in (404, 422):
        return LlmSend(LLM_UNSUPPORTED, body=body, http_status=status_code,
                       error=f"LLM HTTP {status_code}: {snippet}")
    if 400 <= status_code < 500:
        return LlmSend(LLM_REFUSED, body=body, http_status=status_code,
                       error=f"LLM HTTP {status_code}: {snippet}",
                       file_unavailable=_is_file_unavailable(body_text))
    if not (200 <= status_code < 300) or body is None:
        return LlmSend(LLM_UNKNOWN, body=body, http_status=status_code,
                       error=f"LLM HTTP {status_code}, unparseable body: {snippet}")
    code = body.get("code")
    if isinstance(code, int) and not (200 <= code < 300):
        err = _envelope_error(body) or snippet
        if code >= 500 or code == 429:
            return LlmSend(LLM_UNAVAILABLE, body=body, http_status=status_code, error=err[:500])
        if code in (404, 422):
            return LlmSend(LLM_UNSUPPORTED, body=body, http_status=status_code, error=err[:500])
        return LlmSend(LLM_REFUSED, body=body, http_status=status_code, error=err[:500],
                       file_unavailable=_is_file_unavailable(str(body.get("msg"))))
    content = extract(body)
    if not content:
        return LlmSend(LLM_UNKNOWN, body=body, http_status=status_code,
                       error=f"LLM response without content: {snippet}")
    return LlmSend(LLM_ACCEPTED, content=content, body=body, http_status=status_code)


_FILE_UNAVAILABLE_MARKER = "failed to get the file information"


def _is_file_unavailable(text: Optional[str]) -> bool:
    return bool(text) and _FILE_UNAVAILABLE_MARKER in text.lower()


def classify_create_response(status_code: int, body_text: str) -> SendOutcome:
    """Classify one COMPLETE createTask HTTP response (design §7.2).

    * 200 with a ``taskId`` -> accepted;
    * 4xx -> refused; 200 whose body ``code`` is not 200 and has no ``taskId``
      -> refused (kie.ai's HTTP-200 error envelope: nothing was created);
    * 5xx, a non-JSON body, or a 200/2xx without a ``taskId`` and without an
      error code -> unknown.
    Pure, so the matrix is testable without HTTP.
    """
    try:
        data = json.loads(body_text) if body_text else None
    except ValueError:
        data = None
    body = data if isinstance(data, dict) else None
    snippet = (body_text or "")[:300]
    if status_code >= 500:
        return SendOutcome(OUTCOME_UNKNOWN, body=body, http_status=status_code,
                           error=f"createTask HTTP {status_code}: {snippet}")
    if 400 <= status_code < 500:
        return SendOutcome(OUTCOME_REFUSED, body=body, http_status=status_code,
                           error=f"createTask HTTP {status_code}: {snippet}")
    if not (200 <= status_code < 300) or body is None:
        return SendOutcome(OUTCOME_UNKNOWN, body=body, http_status=status_code,
                           error=f"createTask HTTP {status_code}, unparseable body: {snippet}")
    task_id = _extract_task_id(body)
    if task_id:
        return SendOutcome(OUTCOME_ACCEPTED, task_id=str(task_id), body=body,
                           http_status=status_code)
    code = body.get("code")
    if isinstance(code, int) and code != 200:
        return SendOutcome(OUTCOME_REFUSED, body=body, http_status=status_code,
                           error=_no_task_id_msg(body)[:500])
    return SendOutcome(OUTCOME_UNKNOWN, body=body, http_status=status_code,
                       error=_no_task_id_msg(body)[:500])


def classify_chat_response(status_code: int, body_text: str) -> SendOutcome:
    """Classify one COMPLETE chat/completions HTTP response (design §5.2).

    * 2xx with message content -> accepted (``content`` + ``body``);
    * 4xx, or 2xx whose body ``code`` is a non-2xx error -> refused
      (``file_unavailable`` set for Gemini's "Failed to get the file
      information");
    * 5xx, an unparseable body, or a 2xx with neither content nor an error
      code -> unknown (tokens may have been billed).
    """
    try:
        data = json.loads(body_text) if body_text else None
    except ValueError:
        data = None
    body = data if isinstance(data, dict) else None
    snippet = (body_text or "")[:300]
    if status_code >= 500:
        return SendOutcome(OUTCOME_UNKNOWN, body=body, http_status=status_code,
                           error=f"LLM HTTP {status_code}: {snippet}")
    if 400 <= status_code < 500:
        return SendOutcome(OUTCOME_REFUSED, body=body, http_status=status_code,
                           error=f"LLM HTTP {status_code}: {snippet}",
                           file_unavailable=_is_file_unavailable(body_text))
    if not (200 <= status_code < 300) or body is None:
        return SendOutcome(OUTCOME_UNKNOWN, body=body, http_status=status_code,
                           error=f"LLM HTTP {status_code}, unparseable body: {snippet}")
    err = _envelope_error(body)
    if err:
        return SendOutcome(OUTCOME_REFUSED, body=body, http_status=status_code,
                           error=err[:500],
                           file_unavailable=_is_file_unavailable(str(body.get("msg"))))
    content = _extract_chat_content(body)
    if content is None:
        return SendOutcome(OUTCOME_UNKNOWN, body=body, http_status=status_code,
                           error=f"LLM response without message content: {snippet}")
    return SendOutcome(OUTCOME_ACCEPTED, content=content, body=body,
                       http_status=status_code)


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


def _envelope_unavailable(data) -> KieModelUnavailableError | None:
    """Classify an error envelope that another attempt or model could fix.

    ``code >= 500`` or ``429`` is transient (retryable); ``422`` means kie.ai
    does not route this model at the moment and ``404`` ("Invalid model name")
    that it no longer knows it — neither is retryable, but a different model
    may work. Anything else — including success — returns None.
    """
    if not isinstance(data, dict):
        return None
    code = data.get("code")
    if not isinstance(code, int):
        return None
    if code >= 500 or code == 429:
        return KieModelUnavailableError(_envelope_error(data), retryable=True)
    if code in (404, 422):
        return KieModelUnavailableError(_envelope_error(data))
    return None


def chat_credits_consumed(data) -> float | None:
    """kie.ai's own cost of one chat/completions call, or None.

    kie.ai adds a top-level ``credits_consumed`` to an accepted chat body
    (AICP-382; every one of the 130 stored rebuild responses carries it, e.g.
    ``0.25`` for a storyboard shot call, ``~2`` for the global call). It is the
    figure the balance actually moves by — token usage alone does not give it
    (a least-squares fit of the stored ``usage`` against it misses by up to
    60%, hidden media / reasoning tokens). Anything that is not a finite,
    non-negative number (absent, a bool, a non-numeric string, NaN, negative)
    is None: the caller then keeps its conservative estimate.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("credits_consumed")
    if raw is None or isinstance(raw, bool):
        return None
    if not isinstance(raw, (int, float, str)):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return None
    return value


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
    if isinstance(exc, KieModelUnavailableError):
        return exc.retryable
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
            KieUploadError: If the file is over the host's size ceiling, on
                HTTP errors, or on a missing URL in the response.
        """
        path = Path(local_path)
        fname = file_name or path.name
        url = f"{self._upload_base}/api/file-stream-upload"

        # Pre-flight, before the body is opened: the host rejects an oversized
        # upload with a bare 500 that says nothing about size (_UPLOAD_MAX_BYTES).
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise KieUploadError(f"Upload failed: cannot read {local_path}: {exc}") from exc
        if _UPLOAD_MAX_BYTES > 0 and size > _UPLOAD_MAX_BYTES:
            raise KieUploadError(
                f"{fname} is {size / 1_000_000:.1f} MB, over the kie.ai upload "
                f"limit of {_UPLOAD_MAX_BYTES / 1_000_000:.0f} MB — shrink it "
                "before uploading (see media.make_llm_proxy / enforce_size_limit)"
            )

        logger.info(
            "Uploading %s (%.1f MB) to kie.ai (%s/%s)",
            path, size / 1_000_000, upload_path, fname,
        )

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
        resolution: str = "720p",
        aspect_ratio: str = "9:16",
        duration: int,
        model: str = "bytedance/seedance-2",
        generate_audio: bool | None = None,
        reference_audio_urls: list[str] | None = None,
        once: bool = False,
    ) -> str:
        """
        Create a Seedance task on the jobs API.

        ``once=True`` (AICP-351): the same validation and payload, sent with ONE
        POST (:meth:`create_task_once`, no tenacity) and classified — raises
        :class:`KieTaskRefused` (nothing billed) or :class:`KieTaskUnknown`
        (maybe billed) instead of retrying.

        Args:
            prompt: Text prompt for the generation.
            reference_image_urls: Up to 9 image URLs.
            reference_video_urls: Up to 3 video URLs (mp4/mov), each no longer
                than the model's per-clip ceiling — 15 s for Seedance 2.0,
                30 s for Seedance 2.5.
            resolution: ``480p``, ``720p``, or ``1080p``. Which of those a
                given model actually accepts is per-model — the 2.0 Fast/Mini
                variants stop at ``720p``, base 2.0 and 2.5 go to ``1080p`` —
                so callers pass ``ai_models.resolution_or_default`` output
                rather than a raw run value (see :mod:`app.ai_models`).
            aspect_ratio: One of ``1:1|4:3|16:9|9:16|21:9|adaptive``.
            duration: Integer output seconds. The accepted range is per-model:
                [4, 15] for the Seedance 2.0 family, [4, 30] for Seedance 2.5.
            model: kie.ai model id, e.g. ``bytedance/seedance-2``,
                ``bytedance/seedance-2-fast``, ``bytedance/seedance-2-mini``, or
                ``bytedance/seedance-2-5``. Defaults to the base Seedance 2.0
                model.
            generate_audio: Seedance 2.5 only — ask the model to generate (True)
                or suppress (False) an audio track. Left as ``None`` the key is
                omitted from the payload entirely, which is required for the 2.0
                family: those models reject the field.
            reference_audio_urls: Seedance 2.5 only — voice sample URLs whose
                timbre the generated speech should take, in place of the
                arbitrary voice the model otherwise invents. Each file must be
                wav/mp3, 2-30 s and <=15 MB, with a combined length <=30 s;
                ``app.media.make_voice_reference`` guarantees all of that.
                ``None``/empty omits the key, which the 2.0 family requires.

        Returns:
            The ``taskId`` string.

        Raises:
            ValueError: If *duration* is outside the range this *model* accepts,
                or if *generate_audio* / *reference_audio_urls* is passed for a
                model whose spec does not support it.
            KieTaskError: On HTTP or API errors.
        """
        # Validate against the registry entry behind this kie id rather than a
        # hardcoded range — 2.0 and 2.5 disagree on the ceiling.
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

        # Same boundary rule for the voice reference, and same reason: a model
        # without the field rejects the whole request, and finding that out from
        # an opaque kie.ai 4xx costs a cut-and-uploaded clip. Enforce the COUNT
        # here too — the API caps it, and silently truncating would hand back a
        # generation the operator did not ask for.
        if reference_audio_urls:
            spec = ai_models.spec_for(model_key)
            if not spec.max_reference_audios:
                supported = sorted(ai_models.models_supporting_reference_audio())
                raise ValueError(
                    f"reference_audio_urls is not supported by model {model!r} "
                    f"({spec.key}); only {supported} accept it"
                )
            if len(reference_audio_urls) > spec.max_reference_audios:
                raise ValueError(
                    f"model {model!r} accepts at most "
                    f"{spec.max_reference_audios} reference audio url(s), got "
                    f"{len(reference_audio_urls)}"
                )

        url = f"{self._jobs_base}/api/v1/jobs/createTask"
        payload = build_seedance_payload(
            prompt=prompt, reference_image_urls=reference_image_urls,
            reference_video_urls=reference_video_urls, resolution=resolution,
            aspect_ratio=aspect_ratio, duration=duration, model=model,
            generate_audio=generate_audio, reference_audio_urls=reference_audio_urls,
        )
        logger.info(
            "Creating Seedance task (model=%s, resolution=%s, duration=%ds, "
            "generate_audio=%s, reference_audios=%d)",
            model, resolution, duration, generate_audio,
            len(reference_audio_urls or []),
        )

        if once:
            return self._create_once(payload, what="create_task")
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
        once: bool = False,
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
            duration: Output length in seconds — one of 4, 6, 8, 10 (the
                canonical set is ``AI_MODELS["gemini-omni"].allowed_durations``).
            seed: Optional reproducibility seed.

        Returns:
            The ``taskId`` string.

        Raises:
            ValueError: If *duration* is not one of the allowed values.
            KieTaskError: On HTTP or API errors.
        """
        # Same registry-driven validation as create_task: the allowed set lives
        # on AI_MODELS["gemini-omni"].allowed_durations, so this method cannot
        # drift from it the way a local _OMNI_DURATIONS tuple did.
        ai_models.validate_duration("gemini-omni", duration)

        url = f"{self._jobs_base}/api/v1/jobs/createTask"
        payload = build_omni_payload(
            prompt=prompt, image_urls=image_urls, video_url=video_url,
            video_start=video_start, video_end=video_end, resolution=resolution,
            aspect_ratio=aspect_ratio, duration=duration, seed=seed,
        )
        logger.info(
            "Creating Gemini Omni task (resolution=%s, duration=%ds)",
            resolution, duration,
        )

        if once:
            return self._create_once(payload, what="create_omni_task")
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

    def create_image_task(
        self,
        *,
        prompt: str,
        input_urls: list[str],
        aspect_ratio: str = "auto",
        resolution: str | None = None,
        model: str | None = None,
        once: bool = False,
    ) -> str:
        """
        Create a GPT Image (image-to-image) task on the shared jobs API.

        Used by the optional Run preprocessing phase (see
        docs/gpt-image-2-preprocess.md) to transform reference images before
        they reach Seedance/Gemini Omni. Uses the same ``createTask`` /
        ``recordInfo`` protocol as Seedance and Gemini Omni, so polling and
        download reuse :meth:`get_task` and :meth:`download_result`.

        **The model is a setting, not a constant.** ``gpt-image-2``,
        ``gpt-image-2-5-flare`` and ``gpt-image-2-5-sunburst`` accept a
        byte-identical request (same five ``input`` fields, no new or newly
        required params) and return a byte-identical response, so the variant is
        read from ``settings.KIE_IMAGE_MODEL`` at call time and a swap or a
        rollback needs no code change — only an .env edit and a
        ``docker compose up -d`` recreate (``restart`` alone keeps the old
        ``env_file`` values). Both defaults are resolved here rather than in the
        signature for the same reason — a module-level default would freeze the
        env value at import, which would also defeat that recreate for any
        worker process that had already imported this module.

        Args:
            prompt: Text prompt describing the desired image transformation.
                Max 20000 chars per the kie.ai docs.
            input_urls: Public URLs of the source images (max 16 images, 30MB
                each; jpeg/png/webp). Produce these with :meth:`upload_file` or
                pass already-public https URLs.
            aspect_ratio: On the 2.5 variants, one of ``auto|1:1|3:2|2:3|4:3|
                3:4|16:9|9:16|21:9|27:16|16:27|9:8|8:9``. Defaults to ``auto``.
                (gpt-image-2 took a different 16-value enum: it also accepted
                ``5:4|4:5|2:1|1:2|3:1|1:3|9:21``, which 2.5 dropped. Every
                caller sends ``auto``, which all three accept.)
            resolution: ``1K``, ``2K``, or ``4K``; defaults to
                ``settings.KIE_IMAGE_RESOLUTION``. On 2.5 the ``27:16|16:27|
                9:8|8:9`` ratios are 1K-only. On gpt-image-2, ``auto`` +
                2K/4K was a guaranteed create-failure; 2.5 lifted that
                (verified live 2026-09-12), which is what makes a 2K reference
                reachable at all — at 1K a 9:16 frame comes back 941px wide,
                below a 1080p source.
            model: Override the kie.ai model id for this one task. Defaults to
                ``settings.KIE_IMAGE_MODEL``.

        Returns:
            The ``taskId`` string.

        Raises:
            KieTaskError: On HTTP or API errors, or if the response carries no
                ``taskId``.
        """
        url = f"{self._jobs_base}/api/v1/jobs/createTask"
        payload = build_image_payload(
            prompt=prompt, input_urls=input_urls, aspect_ratio=aspect_ratio,
            resolution=resolution, model=model,
        )
        model = payload["model"]
        resolution = payload["input"]["resolution"]
        logger.info(
            "Creating GPT Image task (model=%s, aspect_ratio=%s, resolution=%s, "
            "inputs=%d)", model, aspect_ratio, resolution, len(input_urls),
        )

        if once:
            return self._create_once(payload, what="create_image_task")
        try:
            data = self._create_task_with_retry(url, payload)
        except KieTaskError:
            raise
        except Exception as exc:
            raise KieTaskError(f"create_image_task failed: {exc}") from exc

        task_id = _extract_task_id(data)
        if not task_id:
            raise KieTaskError(_no_task_id_msg(data))

        logger.info("GPT Image task created: model=%s taskId=%s", model, task_id)
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
            KieModelUnavailableError: The model is down (a 5xx/429, HTTP or
                HTTP-200 envelope, that outlived the retries) or not routed
                (422) — another model may still answer.
            KieChatError: On other HTTP errors, on kie.ai's HTTP-200 error
                envelope, or when the response carries no text content.
        """
        url = f"{self._jobs_base}/{model}/v1/chat/completions"
        payload: dict = {"messages": messages, "stream": False}
        if response_format is not None:
            payload["response_format"] = response_format
        logger.info(
            "chat_completion (model=%s, messages=%d, json_schema=%s)",
            model, len(messages), response_format is not None,
        )

        data = self._call_llm_route(
            "chat_completion", model, url, payload, timeout_sec
        )

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
        images: Sequence[tuple[str, bytes]] | None = None,
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
            images: ``(caption, jpeg_bytes)`` pairs appended after the text,
                each as an ``input_text`` caption then an ``input_image`` with a
                base64 ``data:`` URL (verified live 2026-10-20 on gpt-6-1-sol
                and gpt-6-luna; no upload host round-trip needed).
            reasoning_effort: ``low`` | ``medium`` | ``high`` | ``xhigh``.
            timeout_sec: Per-request timeout; defaults to the client's own 60s.

        Returns:
            The assistant message text (all ``output_text`` parts concatenated).

        Raises:
            KieChatError: On HTTP errors, on kie.ai's HTTP-200 error envelope,
                or when the response carries no ``output_text``.
        """
        url = f"{self._jobs_base}/codex/v1/responses"
        content: list[dict] = [{"type": "input_text", "text": input_text}]
        for caption, jpeg in images or ():
            content.append({"type": "input_text", "text": caption})
            content.append({
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,"
                + base64.b64encode(jpeg).decode("ascii"),
            })
        payload = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": reasoning_effort},
            "stream": False,
        }
        logger.info(
            "create_response (model=%s, effort=%s, chars=%d, images=%d)",
            model, reasoning_effort, len(input_text), len(images or ()),
        )

        data = self._call_llm_route(
            "create_response", model, url, payload, timeout_sec
        )

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

    # ------------------------------------------------------------------
    # Single-attempt calls (rebuild paid-call protocol, design §7.2)
    #
    # ONE POST, no tenacity: a request kie.ai accepted whose response was lost
    # must never be sent again from inside the client. The caller records the
    # outcome on its paid-call row; "unknown" is final for that row.
    # ------------------------------------------------------------------

    def create_task_once(self, payload: dict, *, timeout: float) -> SendOutcome:
        """POST *payload* to createTask exactly once and classify the outcome.

        Never raises for a provider or transport problem — every one of them
        is an outcome (:func:`classify_create_response`; a transport error or
        a timeout is ``unknown``: the request may have left). Raises only
        ``ValueError`` for a payload that is not a dict with a ``model``.
        """
        if not isinstance(payload, dict) or not payload.get("model"):
            raise ValueError("create_task_once needs a payload dict with a 'model'")
        url = f"{self._jobs_base}/api/v1/jobs/createTask"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        logger.info("create_task_once (model=%s, timeout=%.0fs)", payload["model"], timeout)
        try:
            resp = self._client.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception as exc:  # transport error / timeout: may have been sent
            logger.warning("create_task_once (model=%s): no complete response: %r",
                           payload["model"], exc)
            return SendOutcome(OUTCOME_UNKNOWN, error=f"no complete response: {exc!r}"[:500])
        outcome = classify_create_response(resp.status_code, resp.text)
        logger.info("create_task_once (model=%s): %s task_id=%s http=%s",
                    payload["model"], outcome.kind, outcome.task_id, outcome.http_status)
        return outcome

    def _create_once(self, payload: dict, *, what: str) -> str:
        """One createTask POST for the ``once=True`` path of the create methods
        (AICP-351 §5.1): the task id, or :class:`KieTaskRefused` /
        :class:`KieTaskUnknown`."""
        outcome = self.create_task_once(payload, timeout=CREATE_ONCE_TIMEOUT_SEC)
        if outcome.accepted and outcome.task_id:
            logger.info("%s (once): taskId=%s", what, outcome.task_id)
            return outcome.task_id
        if outcome.refused:
            raise KieTaskRefused(f"{what} refused: {outcome.error}")
        raise KieTaskUnknown(f"{what} outcome unknown: {outcome.error}")

    def chat_completion_once(
        self,
        model: str,
        messages: list[dict],
        *,
        timeout: float,
        response_format: dict | None = None,
    ) -> SendOutcome:
        """One chat/completions POST (no ``_chat_with_retry``), classified by
        :func:`classify_chat_response`. Never raises for a provider or
        transport problem."""
        url = f"{self._jobs_base}/{model}/v1/chat/completions"
        payload: dict = {"messages": messages, "stream": False}
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        logger.info("chat_completion_once (model=%s, messages=%d, timeout=%.0fs)",
                    model, len(messages), timeout)
        try:
            resp = self._client.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception as exc:
            logger.warning("chat_completion_once (model=%s): no complete response: %r",
                           model, exc)
            return SendOutcome(OUTCOME_UNKNOWN, error=f"no complete response: {exc!r}"[:500])
        outcome = classify_chat_response(resp.status_code, resp.text)
        logger.info("chat_completion_once (model=%s): %s http=%s",
                    model, outcome.kind, outcome.http_status)
        return outcome

    def _llm_post_once(self, url: str, payload: dict, timeout: float, extract) -> LlmSend:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._client.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception as exc:  # transport error / timeout: may have been sent
            return LlmSend(LLM_UNKNOWN, error=f"no complete response: {exc!r}"[:500])
        return classify_llm_response(resp.status_code, resp.text, extract)

    def chat_completion_send_once(
        self, *, model: str, messages: list[dict], response_format: dict | None = None,
        timeout_sec: float,
    ) -> LlmSend:
        """:meth:`chat_completion` with ONE POST and no retry (AICP-351 §5.2):
        the caller claimed this call's permit and draw first."""
        url = f"{self._jobs_base}/{model}/v1/chat/completions"
        payload: dict = {"messages": messages, "stream": False}
        if response_format is not None:
            payload["response_format"] = response_format
        out = self._llm_post_once(url, payload, timeout_sec, _extract_chat_content)
        logger.info("chat_completion_send_once (model=%s): %s http=%s", model, out.kind,
                    out.http_status)
        return out

    def create_response_send_once(
        self, *, model: str, input_text: str,
        images: Sequence[tuple[str, bytes]] | None = None,
        reasoning_effort: str = "medium", timeout_sec: float,
    ) -> LlmSend:
        """:meth:`create_response` with ONE POST and no retry (AICP-351 §5.2)."""
        url = f"{self._jobs_base}/codex/v1/responses"
        content: list[dict] = [{"type": "input_text", "text": input_text}]
        for caption, jpeg in images or ():
            content.append({"type": "input_text", "text": caption})
            content.append({
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,"
                + base64.b64encode(jpeg).decode("ascii"),
            })
        payload = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": reasoning_effort},
            "stream": False,
        }
        out = self._llm_post_once(url, payload, timeout_sec, _extract_response_output_text)
        logger.info("create_response_send_once (model=%s): %s http=%s", model, out.kind,
                    out.http_status)
        return out

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

    def get_credit_balance(self) -> float:
        """
        Fetch the account's remaining credit balance.

        Calls the kie.ai common API ``GET /api/v1/chat/credit``, whose ``data``
        field is the remaining credit quantity.

        Returns:
            The remaining credits as a float.

        Raises:
            KieError: On HTTP errors or an unexpected response structure.
        """
        url = f"{self._jobs_base}/api/v1/chat/credit"
        try:
            data = self._get_credit_with_retry(url)
        except KieError:
            raise
        except Exception as exc:
            raise KieError(f"get_credit_balance failed: {exc}") from exc

        balance = data.get("data") if isinstance(data, dict) else None
        if balance is None:
            raise KieError(f"credit response missing 'data': {data}")
        try:
            return float(balance)
        except (ValueError, TypeError) as exc:
            raise KieError(f"credit balance is not numeric: {balance!r}") from exc

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
        Stream-download a result file (video or image) to a local path.

        Downloads land on a sibling ``.part`` file and are moved into place only
        once the body is complete, so a caller never finds a truncated file at
        *dst_path*.  Transient network failures are retried (see
        :func:`_transient_retry`); this matters because the result CDN
        intermittently closes a connection mid-body, which httpx reports as
        ``RemoteProtocolError``.  Before the retry existed, one such truncation
        failed the whole run over an image that was sitting on the CDN intact
        (run ce9b5bfd, 2026-08-27) — and a preprocess run that fails this way
        cannot be retried at all, because the empty slot crashes the submit.

        Args:
            url: The result URL (from :meth:`poll_task`).
            dst_path: Local destination path.

        Raises:
            KieError: On HTTP errors, or when the body is short after retries.
        """
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading result %s → %s", url, dst)

        part = dst.with_name(dst.name + ".part")
        try:
            self._download_to_part(url, part)
            os.replace(part, dst)
        finally:
            # A failed attempt must not leave a stray .part behind; the retry
            # decorator reopens it from scratch anyway.
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    logger.warning("Could not unlink partial download %s", part)

        logger.info("Download complete: %s (%d bytes)", dst, dst.stat().st_size)

    @_transient_retry
    def _download_to_part(self, url: str, part: Path) -> None:
        """Stream *url* into *part*, verifying the body arrived whole.

        Retried by ``_transient_retry`` on network errors and 5xx.  A body
        shorter than the advertised ``Content-Length`` raises
        ``httpx.RemoteProtocolError`` from httpx itself; we additionally check
        the byte count, because a CDN that reports a length and then closes
        cleanly would otherwise hand us a silently truncated file.
        """
        with self._client.stream("GET", url) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500:
                    raise  # transient — let the retry decorator see it
                raise KieResultGone(
                    f"Download failed with status {exc.response.status_code}: {url}"
                ) from exc
            expected = resp.headers.get("content-length")
            written = 0
            with part.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    fh.write(chunk)
                    written += len(chunk)

        if expected is not None:
            try:
                expected_n = int(expected)
            except ValueError:
                expected_n = None
            if expected_n is not None and written != expected_n:
                # Transport-level truncation the transport did not flag.
                raise httpx.RemoteProtocolError(
                    f"incomplete download: received {written} bytes, "
                    f"expected {expected_n}"
                )

    def close(self) -> None:
        """Close the underlying HTTP client and free its connection pool."""
        self._client.close()

    def __enter__(self) -> "KieClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal retry-wrapped helpers
    # ------------------------------------------------------------------

    @_transient_retry
    def _upload_with_retry(self, url: str, fh, upload_path: str, fname: str) -> dict:
        fh.seek(0)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        resp = self._client.post(
            url,
            headers=headers,
            files={"file": (fname, fh, "application/octet-stream")},
            data={"uploadPath": upload_path, "fileName": fname},
        )
        if resp.status_code >= 500:
            # raise_for_status() builds its message from the status line alone,
            # so without this line the host's own explanation of a 5xx is never
            # recorded anywhere — which is how a hard size rejection spent a day
            # looking like a flaky upstream.
            logger.warning(
                "kie.ai upload HTTP %s for %s: %s",
                resp.status_code, fname, (resp.text or "<empty body>")[:300],
            )
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

    def _call_llm_route(
        self, what: str, model: str, url: str, payload: dict,
        timeout_sec: float | None,
    ) -> dict:
        """POST to an LLM route with retries; return a body that is not an error.

        Raises :class:`KieModelUnavailableError` (prefixed ``"{what} {model}: "``)
        when another model might answer — see its docstring — and
        :class:`KieChatError` otherwise.
        """
        try:
            data = self._chat_with_retry(url, payload, timeout_sec)
        except KieModelUnavailableError as exc:
            raise KieModelUnavailableError(f"{what} {model}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            # Only a 5xx reaches here (_chat_with_retry turns a 4xx into a
            # KieChatError), and only once the retries are spent.
            raise KieModelUnavailableError(
                f"{what} {model}: HTTP {exc.response.status_code}"
            ) from exc
        except KieChatError:
            raise
        except Exception as exc:
            raise KieChatError(f"{what} failed: {exc}") from exc

        unavailable = _envelope_unavailable(data)
        if unavailable is not None:
            raise KieModelUnavailableError(f"{what} {model}: {unavailable}")
        err = _envelope_error(data)
        if err:
            raise KieChatError(f"{what} {model}: {err}")
        return data

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
            data = resp.json()
        except ValueError as exc:
            raise KieChatError(
                f"LLM response is not JSON: {resp.text[:300]}"
            ) from exc
        # The 5xx envelope arrives as HTTP 200, so the status check above never
        # sees it; raising here is what puts it under the retry decorator.
        unavailable = _envelope_unavailable(data)
        if unavailable is not None and unavailable.retryable:
            raise unavailable
        return data

    @_transient_retry
    def _get_credit_with_retry(self, url: str) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        resp = self._client.get(url, headers=headers)
        if resp.status_code >= 500:
            resp.raise_for_status()
        if not resp.is_success:
            raise KieError(
                f"credit HTTP {resp.status_code}: {resp.text[:300]}"
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


# ---------------------------------------------------------------------------
# Module-level shared instance
# ---------------------------------------------------------------------------

_shared_client: KieClient | None = None
_shared_client_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request payload builders (AICP-299 D6)
#
# Pure functions: the exact JSON body ``createTask`` receives, minus auth. The
# ``create_*`` methods validate and then call these, and knowledge capture
# records their output as the attempt's request, so the recorded request is
# byte-identical to what was sent.
# ---------------------------------------------------------------------------


def build_seedance_payload(
    *,
    prompt: str,
    reference_image_urls: list[str],
    reference_video_urls: list[str],
    resolution: str = "720p",
    aspect_ratio: str = "9:16",
    duration: int,
    model: str = "bytedance/seedance-2",
    generate_audio: bool | None = None,
    reference_audio_urls: list[str] | None = None,
) -> dict:
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
    # Same rule for the voice reference — absent unless actually supplied.
    if reference_audio_urls:
        payload["input"]["reference_audio_urls"] = list(reference_audio_urls)
    return payload


def build_omni_payload(
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
) -> dict:
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
    return payload


def build_image_payload(
    *,
    prompt: str,
    input_urls: list[str],
    aspect_ratio: str = "auto",
    resolution: str | None = None,
    model: str | None = None,
) -> dict:
    """Model and resolution default to the settings read at call time (see
    ``KieClient.create_image_task``)."""
    return {
        "model": model or settings.KIE_IMAGE_MODEL,
        "input": {
            "prompt": prompt,
            "input_urls": input_urls,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution or settings.KIE_IMAGE_RESOLUTION,
        },
    }


def build_kling_payload(
    *,
    prompt: str,
    image_urls: list[str],
    duration: int,
    resolution: str = "720p",
    aspect_ratio: str = "9:16",
) -> dict:
    """Kling 4.0 reference-to-video (images-only rebuild chunk).

    Pilot-verified shape (2026-11-02): an integer ``duration`` (3-30), at most
    ``max_reference_images`` (10) images, addressed in the prompt as
    ``@image_N``. Validates against the registry (raises ``ValueError``)."""
    spec = ai_models.AI_MODELS["kling-4-0"]
    ai_models.validate_images_only_duration(spec.key, duration)
    _check_images_only_inputs(spec, image_urls, resolution, aspect_ratio)
    return {
        "model": spec.kie_model_id,
        "input": {
            "prompt": prompt,
            "duration": int(duration),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "image_urls": list(image_urls),
        },
    }


def build_seedance_images_payload(
    *,
    prompt: str,
    reference_image_urls: list[str],
    duration: int,
    resolution: str = "720p",
    aspect_ratio: str = "9:16",
    model: str = "seedance-2-5",
) -> dict:
    """Seedance images-only request (no reference video), pilot-verified for
    2.5: ``reference_video_urls: []``, an integer ``duration``, a CONCRETE
    aspect ratio and ``generate_audio: false``. *model* is the registry key.
    :func:`build_seedance_payload` (the v2v builder) is untouched."""
    spec = ai_models.AI_MODELS.get(model)
    if spec is None or not spec.supports_images_only or spec.family != "seedance":
        raise ValueError(f"model {model!r} has no Seedance images-only mode")
    ai_models.validate_images_only_duration(spec.key, duration)
    _check_images_only_inputs(spec, reference_image_urls, resolution, aspect_ratio)
    payload = {
        "model": spec.kie_model_id,
        "input": {
            "prompt": prompt,
            "reference_image_urls": list(reference_image_urls),
            "reference_video_urls": [],
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        },
    }
    if spec.supports_generate_audio:
        payload["input"]["generate_audio"] = False
    return payload


def build_images_only_payload(
    *,
    model: str,
    prompt: str,
    image_urls: list[str],
    duration: int,
    resolution: str,
    aspect_ratio: str,
) -> dict:
    """Dispatch on the registry family: the one builder an images-only
    attempt's send step calls."""
    spec = ai_models.AI_MODELS.get(model)
    if spec is None or not spec.supports_images_only:
        raise ValueError(f"model {model!r} has no images-only mode")
    if spec.family == "kling":
        return build_kling_payload(prompt=prompt, image_urls=image_urls, duration=duration,
                                   resolution=resolution, aspect_ratio=aspect_ratio)
    if spec.family == "seedance":
        return build_seedance_images_payload(
            prompt=prompt, reference_image_urls=image_urls, duration=duration,
            resolution=resolution, aspect_ratio=aspect_ratio, model=model,
        )
    raise ValueError(f"no images-only payload builder for family {spec.family!r}")


def _check_images_only_inputs(spec, image_urls, resolution, aspect_ratio) -> None:
    if not image_urls:
        raise ValueError(f"{spec.key}: an images-only request needs at least one image")
    if len(image_urls) > spec.max_reference_images:
        raise ValueError(
            f"{spec.key} accepts at most {spec.max_reference_images} images, "
            f"got {len(image_urls)}"
        )
    if resolution not in spec.resolutions:
        raise ValueError(f"{spec.key} has no {resolution!r} resolution")
    if aspect_ratio not in ("9:16", "16:9", "1:1"):
        raise ValueError(
            f"images-only aspect_ratio must be concrete (9:16 / 16:9 / 1:1), "
            f"got {aspect_ratio!r}"
        )


def get_shared_client() -> KieClient:
    """Return a lazily-created, module-level KieClient (thread-safe)."""
    global _shared_client
    if _shared_client is None:
        with _shared_client_lock:
            if _shared_client is None:
                _shared_client = KieClient()
    return _shared_client
