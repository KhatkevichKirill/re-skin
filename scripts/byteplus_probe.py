#!/usr/bin/env python3
"""
byteplus_probe.py — answer the open questions in
wiki/decisions/byteplus-second-provider.md against a real Ark account.

This is deliberately a standalone script with no app imports: it runs on the
host before any of the integration exists, and its job is to decide the SHAPE of
that integration. What it learns gets folded back into the decision record.

Staging is by cost, not by topic. Everything up to and including stage 2 is free
or ~$0.04; stage 3 generates real video and bills real money, so it is gated
behind BOTH --paid and --yes.

    python3 scripts/byteplus_probe.py                 # stages 0-2 (free)
    python3 scripts/byteplus_probe.py --paid --yes    # everything

Reads ARK_API_KEY and ARK_BASE_URL from the environment or ./.env. The key is
never printed.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socket
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
REPORT = Path(os.environ.get("PROBE_REPORT", "/tmp/byteplus-probe-report.json"))

# Model ids verified against GET /api/v3/models on 2026-08-17. NOTE: the
# research doc's "seedream-5-0-lite-260128" does not exist in the catalogue —
# the real ids are below.
SEEDANCE_2_5 = "dreamina-seedance-2-5-260628"
SEEDANCE_2_0 = "dreamina-seedance-2-0-260128"
SEEDREAM_5_0 = "seedream-5-0-260128"
SEEDREAM_5_0_PRO = "dola-seedream-5-0-pro-260628"

results: dict = {}


def load_env() -> tuple[str, str]:
    """Return (api_key, base_url), reading ./.env if the vars are not exported."""
    key = os.environ.get("ARK_API_KEY", "")
    base = os.environ.get("ARK_BASE_URL", "")
    if not key or not base:
        env_path = REPO / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k == "ARK_API_KEY" and not key:
                    key = v.strip()
                elif k == "ARK_BASE_URL" and not base:
                    base = v.strip()
    if not key:
        sys.exit("ARK_API_KEY is not set (export it or put it in .env)")
    return key, (base or "https://ark.ap-southeast.bytepluses.com/api/v3")


def record(name: str, **fields) -> None:
    results[name] = fields
    status = fields.get("status", "?")
    detail = fields.get("detail", "")
    print(f"  [{status:^7}] {name}{': ' + detail if detail else ''}")


# ---------------------------------------------------------------------------
# Stage 0 — auth and activation (free)
# ---------------------------------------------------------------------------

def stage0(client: httpx.Client, base: str) -> None:
    """Auth check + which models this account may actually call.

    `ModelNotOpen` is evaluated BEFORE parameter validation, which is what makes
    this free: a request with an empty `content` array never reaches the
    generator, so an activated model answers with InvalidParameter and an
    unactivated one answers with ModelNotOpen. Neither bills.
    """
    print("\nStage 0 — auth + model activation (free)")

    r = client.get(f"{base}/contents/generations/tasks/cgt-probe-nonexistent")
    ok = r.status_code == 404 and "ResourceNotFound" in r.text
    record(
        "auth",
        status="PASS" if ok else "FAIL",
        detail="key authenticates" if ok else f"HTTP {r.status_code}: {r.text[:200]}",
    )
    if not ok:
        sys.exit("auth failed — stopping")

    for model in (SEEDANCE_2_5, SEEDANCE_2_0, SEEDREAM_5_0, SEEDREAM_5_0_PRO):
        r = client.post(
            f"{base}/contents/generations/tasks",
            json={"model": model, "content": []},
        )
        code = (r.json().get("error") or {}).get("code", "")
        activated = code != "ModelNotOpen"
        record(
            f"activated:{model}",
            status="PASS" if activated else "BLOCKED",
            detail=code or "activated",
        )


def activated(model: str) -> bool:
    entry = results.get(f"activated:{model}", {})
    return entry.get("status") == "PASS"


# ---------------------------------------------------------------------------
# Stage 1 — Seedream protocol shape (free / ~$0.02)
# ---------------------------------------------------------------------------

def stage1(client: httpx.Client, base: str) -> None:
    """Probe #6: is Seedream synchronous, and what does it return?

    Decides how much of the video create/poll/download protocol Track A actually
    rehearses. A synchronous route shares only the client, auth and error
    taxonomy — worth knowing before Phase A is estimated.
    """
    print("\nStage 1 — Seedream t2i protocol (free on online inference)")
    if not activated(SEEDREAM_5_0):
        record("seedream-t2i", status="SKIP", detail="model not activated")
        return

    t0 = time.monotonic()
    r = client.post(
        f"{base}/images/generations",
        json={
            "model": SEEDREAM_5_0,
            # A fictional face: this doubles as the trusted-input persona for
            # stage 3, and the prompt deliberately avoids anything that could
            # trip the separate celebrity/copyright filter.
            "prompt": (
                "studio portrait photograph of a fictional 30-year-old woman, "
                "neutral expression, plain grey background, soft even lighting, "
                "photorealistic, not resembling any real person"
            ),
            "size": "1024x1024",
            "response_format": "url",
        },
        timeout=180.0,
    )
    elapsed = time.monotonic() - t0
    if not r.is_success:
        record("seedream-t2i", status="FAIL", detail=f"HTTP {r.status_code}: {r.text[:250]}")
        return

    body = r.json()
    url = (body.get("data") or [{}])[0].get("url")
    synchronous = url is not None
    record(
        "seedream-t2i",
        status="PASS" if synchronous else "INFO",
        detail=(
            f"synchronous, {elapsed:.1f}s, usage={body.get('usage')}"
            if synchronous
            else f"no url in response — async? keys={list(body)}"
        ),
        image_url=url,
        elapsed_sec=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Stage 2 — will Ark fetch bytes from US? (free)
# ---------------------------------------------------------------------------

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves one file, and records whether anyone actually fetched it.

    Mirrors the real route's shape (a query string carrying a token) so the test
    also covers "does Ark preserve the query string when fetching".
    """

    served: list = []
    payload: bytes = b""
    content_type: str = "image/png"

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        _QuietHandler.served.append(
            {"path": parsed.path, "query": parsed.query, "ua": self.headers.get("User-Agent", "")}
        )
        self.send_response(200)
        self.send_header("Content-Type", _QuietHandler.content_type)
        self.send_header("Content-Length", str(len(_QuietHandler.payload)))
        self.end_headers()
        self.wfile.write(_QuietHandler.payload)

    def log_message(self, *_):
        pass


def stage2(client: httpx.Client, base: str, host_ip: str) -> None:
    """Probe #1: can Ark fetch `http://<bare-ip>:<port>/...?token=...`?

    This is the question that gates BOTH tracks, and it is answered here without
    deploying a line of code: a throwaway server on a spare port stands in for
    the /public/runs/{rid}/file route. If Ark fetches from this, it will fetch
    from nginx; if it refuses, we need a TosPublisher and the design changes.

    Uses Seedream i2i (free) rather than a video generation, so the answer costs
    nothing.
    """
    print("\nStage 2 — will Ark fetch from our host? (free, via Seedream i2i)")
    if not activated(SEEDREAM_5_0):
        record("ark-fetches-us", status="SKIP", detail="seedream not activated")
        return

    src = results.get("seedream-t2i", {}).get("image_url")
    if not src:
        record("ark-fetches-us", status="SKIP", detail="stage 1 produced no image to re-serve")
        return

    _QuietHandler.payload = httpx.get(src, timeout=120.0).content
    _QuietHandler.content_type = "image/png"
    _QuietHandler.served = []

    # Bind a spare port; the box has no firewall (INPUT ACCEPT, no ufw), so a
    # bare-IP plain-HTTP URL here is the same reachability question nginx's 8847
    # poses.
    sock = socket.socket()
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _QuietHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        probe_url = (
            f"http://{host_ip}:{port}/runs/probe/file"
            f"?path=refs/0.png&token={'a' * 64}"
        )
        print(f"    serving {probe_url}")
        r = client.post(
            f"{base}/images/generations",
            json={
                "model": SEEDREAM_5_0,
                "prompt": "change the background to a plain white studio backdrop",
                "image": probe_url,
                "size": "1024x1024",
                "response_format": "url",
            },
            timeout=180.0,
        )
        fetched = bool(_QuietHandler.served)
        record(
            "ark-fetches-us",
            status="PASS" if fetched else "FAIL",
            detail=(
                f"fetched (query preserved: {_QuietHandler.served[0]['query']!r}, "
                f"ua={_QuietHandler.served[0]['ua'][:40]!r})"
                if fetched
                else f"no fetch seen — HTTP {r.status_code}: {r.text[:250]}"
            ),
            http_status=r.status_code,
            response=r.text[:400],
        )
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Stage 3 — video generation (PAID)
# ---------------------------------------------------------------------------

def submit_video(client: httpx.Client, base: str, payload: dict) -> tuple[int, dict]:
    r = client.post(f"{base}/contents/generations/tasks", json=payload, timeout=120.0)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw": r.text[:400]}


def poll_video(client: httpx.Client, base: str, task_id: str, timeout: float = 900.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        r = client.get(f"{base}/contents/generations/tasks/{task_id}", timeout=60.0)
        last = r.json()
        if last.get("status") in ("succeeded", "failed", "cancelled", "expired"):
            return last
        time.sleep(10)
    return {**last, "status": "probe-timeout"}


def stage3(client: httpx.Client, base: str, args) -> None:
    """Probes #2/#3/#4/#5 — the ones that cost money.

    Each is a real generation on a real account. Order matters: the control
    (#3) runs first, because if OUR REQUEST SHAPE is wrong then a failure on
    the go/no-go (#2) proves nothing about moderation.
    """
    print("\nStage 3 — video generation (PAID)")
    if not activated(SEEDANCE_2_5):
        record("video", status="SKIP", detail="seedance 2.5 not activated")
        return
    if not (args.paid and args.yes):
        record("video", status="SKIP", detail="needs --paid --yes (bills real money)")
        return

    persona = results.get("seedream-t2i", {}).get("image_url")
    if not (persona and args.source_video_url):
        record(
            "video",
            status="SKIP",
            detail="needs a persona image (stage 1) and --source-video-url",
        )
        return

    base_payload = {
        "model": SEEDANCE_2_5,
        "omni_reference_task_type": "edit",
        "ratio": "adaptive",
        "duration": -1,
        "resolution": "720p",
        "generate_audio": True,
        "watermark": False,
    }

    def run(name: str, video_url: str, note: str) -> None:
        payload = {
            **base_payload,
            "content": [
                {
                    "type": "text",
                    "text": "Replace the person in @Video 1 with the person in @Image 1. "
                            "Keep the original motion, framing and timing.",
                },
                {"type": "image_url", "image_url": {"url": persona}, "role": "reference_image"},
                {"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"},
            ],
        }
        t0 = time.monotonic()
        code, body = submit_video(client, base, payload)
        submit_sec = time.monotonic() - t0
        task_id = body.get("id")
        if not task_id:
            # The interesting half of probe #2: a SYNCHRONOUS rejection means we
            # learn the answer before being billed.
            record(
                name,
                status="REJECTED-AT-SUBMIT",
                detail=f"{note} — HTTP {code} in {submit_sec:.1f}s: "
                       f"{(body.get('error') or {}).get('code', body)}",
                response=str(body)[:500],
            )
            return
        final = poll_video(client, base, task_id)
        record(
            name,
            status=final.get("status", "?").upper(),
            detail=f"{note} — task={task_id} "
                   f"usage={final.get('usage')} error={final.get('error')}",
            task_id=task_id,
            duration_sec=final.get("duration"),
            video_url=(final.get("content") or {}).get("video_url"),
            raw=final,
        )

    # #3 control first — proves the request shape before #2 tests moderation.
    if args.control_video_url:
        run("video-control", args.control_video_url, "Seedance-generated input (trusted)")
    # #2 the go/no-go.
    run("video-gonogo", args.source_video_url, "real client footage (moderation test)")

    # #4 concurrency: does the 4th+ concurrent submit 429, or queue?
    print("    concurrency: submitting 5 tasks at once")
    codes = []
    for _ in range(5):
        code, body = submit_video(
            client,
            base,
            {
                **base_payload,
                "content": [
                    {"type": "text", "text": "Replace the person in @Video 1 with @Image 1."},
                    {"type": "image_url", "image_url": {"url": persona}, "role": "reference_image"},
                    {"type": "video_url", "video_url": {"url": args.source_video_url},
                     "role": "reference_video"},
                ],
            },
        )
        codes.append((code, body.get("id") or (body.get("error") or {}).get("code")))
    rejected = [c for c in codes if not str(c[1] or "").startswith("cgt-")]
    record(
        "concurrency-5",
        status="QUEUES" if not rejected else "REJECTS",
        detail=f"{len(codes) - len(rejected)}/5 accepted; rejections={rejected}",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paid", action="store_true", help="run stage 3 (bills real money)")
    ap.add_argument("--yes", action="store_true", help="confirm the paid stage")
    ap.add_argument("--source-video-url", help="public URL of REAL client footage (probe #2)")
    ap.add_argument("--control-video-url", help="public URL of a Seedance-generated clip (probe #3)")
    ap.add_argument("--host-ip", default="203.0.113.10", help="our public IP (stage 2)")
    args = ap.parse_args()

    key, base = load_env()
    print(f"Ark base: {base}")
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=60.0,
    ) as client:
        stage0(client, base)
        stage1(client, base)
        stage2(client, base, args.host_ip)
        stage3(client, base, args)

    REPORT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nreport → {REPORT}")
    blocked = [k for k, v in results.items() if v.get("status") in ("FAIL", "BLOCKED")]
    if blocked:
        print(f"blocked/failed: {', '.join(blocked)}")


if __name__ == "__main__":
    main()
