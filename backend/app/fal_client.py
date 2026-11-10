"""Thin HTTP client for Fal (AICP-312 Stage 1, docs/fal-provider-design.md §3, §6).

Transport only: every method makes ONE attempt and returns a small typed
outcome that ``pipeline_provider`` turns into a state change. The paid call
(:meth:`FalClient.submit`) is never retried here — at Fal two identical submits
are two generations (probe T3). Never logs a header; every provider text is
scrubbed of the key before it is returned.

URLs follow one rule (design §3): submit is ``POST {QUEUE_BASE}/{endpoint id}``;
status, result and cancel live under ``{QUEUE_BASE}/{owner}/{app}/requests/{id}``
where owner/app are the endpoint id's first two segments.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

log = logging.getLogger(__name__)

QUEUE_BASE = "https://queue.fal.run"
REST_BASE = "https://rest.fal.ai"
CDN_BASE = "https://v3.fal.media"

# Fal queue statuses (docs). COMPLETED means "finished", not "succeeded": the
# result GET says which (probe T4: a 422 after COMPLETED).
STATUS_COMPLETED = "COMPLETED"
STATUS_PENDING = frozenset({"IN_QUEUE", "IN_PROGRESS"})
# Not documented as statuses, but if Fal ever reports them the request is over.
STATUS_TERMINAL_FAILURE = frozenset({"CANCELLED", "CANCELED", "FAILED", "ERROR"})


def scrub(text: str, key: str) -> str:
    text = text or ""
    if key:
        text = text.replace(key, "***")
    return text[:500]


def request_urls(queue_app: str, request_id: str) -> dict[str, str]:
    base = f"{QUEUE_BASE}/{queue_app}/requests/{request_id}"
    return {"status_url": f"{base}/status", "response_url": base, "cancel_url": f"{base}/cancel"}


@dataclass
class SubmitOutcome:
    # accepted | refused | unknown
    kind: str
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    urls: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class StatusOutcome:
    # ok | transient | auth | gone
    kind: str
    http_status: Optional[int] = None
    fal_status: Optional[str] = None
    error: str = ""


@dataclass
class ResultOutcome:
    # ok | refused | retry
    kind: str
    http_status: Optional[int] = None
    body: dict = field(default_factory=dict)
    category: Optional[str] = None
    error: str = ""


class FalUploadError(Exception):
    pass


class FalDownloadError(Exception):
    pass


def _lifecycle(seconds: int) -> str:
    return json.dumps({"expiration_duration_seconds": int(seconds)})


class FalClient:
    def __init__(self, api_key: str, *, transport: Optional[httpx.BaseTransport] = None):
        if not api_key:
            raise ValueError("FAL_API_KEY is not set")
        self._key = api_key
        self._http = httpx.Client(timeout=60.0, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Key {self._key}"}

    def _scrub(self, text: str) -> str:
        return scrub(text, self._key)

    # -- free calls --------------------------------------------------------

    def upload(self, data: bytes, *, content_type: str, file_name: str,
               lifecycle_sec: int) -> str:
        """Upload bytes to the Fal CDN (token + one POST, as fal-client 1.0.3
        does) and return the access URL. Free; generates nothing."""
        try:
            tok = self._http.post(
                f"{REST_BASE}/storage/auth/token",
                params={"storage_type": "fal-cdn-v3"},
                headers={**self._auth(), "Accept": "application/json",
                         "Content-Type": "application/json"},
                json={},
            )
            tok.raise_for_status()
            t = tok.json()
            resp = self._http.post(
                f"{CDN_BASE}/files/upload",
                headers={
                    "Authorization": f"{t['token_type']} {t['token']}",
                    "Content-Type": content_type,
                    "X-Fal-File-Name": file_name,
                    "X-Fal-Object-Lifecycle": _lifecycle(lifecycle_sec),
                    "X-Fal-Object-Lifecycle-Preference": _lifecycle(lifecycle_sec),
                },
                content=data,
                timeout=300.0,
            )
            resp.raise_for_status()
            url = resp.json()["access_url"]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise FalUploadError(self._scrub(f"{type(exc).__name__}: {exc}")) from None
        if not isinstance(url, str) or not url.startswith("https://"):
            raise FalUploadError("upload answered without an https access_url")
        return url

    def status(self, status_url: str) -> StatusOutcome:
        try:
            r = self._http.get(status_url, headers=self._auth(), timeout=30.0)
        except httpx.HTTPError as exc:
            return StatusOutcome("transient", error=type(exc).__name__)
        if 200 <= r.status_code < 300:
            try:
                st = (r.json() or {}).get("status")
            except ValueError:
                return StatusOutcome("transient", r.status_code, error="unreadable 2xx")
            return StatusOutcome("ok", r.status_code, fal_status=st)
        if r.status_code == 429 or r.status_code >= 500:
            return StatusOutcome("transient", r.status_code, error=self._scrub(r.text))
        if r.status_code in (401, 403):
            return StatusOutcome("auth", r.status_code, error=self._scrub(r.text))
        return StatusOutcome("gone", r.status_code, error=self._scrub(r.text))

    def result(self, response_url: str) -> ResultOutcome:
        try:
            r = self._http.get(response_url, headers=self._auth(), timeout=60.0)
        except httpx.HTTPError as exc:
            return ResultOutcome("retry", error=type(exc).__name__)
        text = self._scrub(r.text)
        if r.status_code == 422:
            cat = "moderation" if "content_policy_violation" in r.text else "invalid"
            return ResultOutcome("refused", 422, category=cat, error=text)
        if r.status_code != 200:
            return ResultOutcome("retry", r.status_code, error=text)
        try:
            body = r.json()
        except ValueError:
            return ResultOutcome("retry", 200, error="unreadable result")
        if not isinstance(body, dict):
            return ResultOutcome("retry", 200, error="result is not an object")
        return ResultOutcome("ok", 200, body=body)

    def cancel(self, cancel_url: str) -> Optional[int]:
        """Idempotent cancel request; returns the HTTP status (None = no answer)."""
        try:
            return self._http.put(cancel_url, headers=self._auth(), timeout=30.0).status_code
        except httpx.HTTPError:
            return None

    def download(self, url: str, dst: str, *, max_bytes: int) -> int:
        """Stream *url* into *dst* (which must not exist); returns its size."""
        size = 0
        try:
            with self._http.stream("GET", url, timeout=300.0) as resp:
                resp.raise_for_status()
                with open(dst, "xb") as fh:
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise FalDownloadError("result larger than the download cap")
                        fh.write(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())
        except httpx.HTTPError as exc:
            raise FalDownloadError(type(exc).__name__) from None
        return size

    # -- the paid call ----------------------------------------------------

    def submit(self, endpoint_id: str, body: dict, *, queue_app: str,
               output_lifecycle_sec: int) -> SubmitOutcome:
        """ONE POST. A clear 4xx is ``refused``; a transport error, a 5xx, an
        unreadable 2xx or a 2xx without a request id is ``unknown`` (the
        request may exist and be billed)."""
        headers = {
            **self._auth(),
            "Content-Type": "application/json",
            "X-Fal-Store-IO": "0",
            "X-Fal-No-Retry": "1",
            "X-Fal-Object-Lifecycle-Preference": _lifecycle(output_lifecycle_sec),
        }
        try:
            r = self._http.post(f"{QUEUE_BASE}/{endpoint_id}", headers=headers,
                                json=body, timeout=60.0)
        except httpx.HTTPError as exc:
            return SubmitOutcome("unknown", error=f"no response ({type(exc).__name__})")
        if 400 <= r.status_code < 500:
            return SubmitOutcome("refused", r.status_code, error=self._scrub(r.text))
        if not 200 <= r.status_code < 300:
            return SubmitOutcome("unknown", r.status_code, error=self._scrub(r.text))
        try:
            data = r.json()
        except ValueError:
            return SubmitOutcome("unknown", r.status_code, error="unreadable 2xx")
        rid = data.get("request_id") if isinstance(data, dict) else None
        if not isinstance(rid, str) or not rid.strip():
            return SubmitOutcome("unknown", r.status_code, error="2xx without request_id")
        # URLs are rebuilt by the §3 rule, never trusted from the body.
        return SubmitOutcome("accepted", r.status_code, request_id=rid,
                             urls=request_urls(queue_app, rid))
