"""
FastAPI application for re-skin.
Wires together the REST API (/api/v2) and the operator web UI (/v2).
"""

import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api_overlay import router as api_overlay_router
from .api_ops import router as api_ops_router, web_router as ops_web_router
from .api_paid_attempts import router as api_paid_attempts_router
from .api_assistant import router as api_assistant_router
from .knowledge.api import router as knowledge_router
from .knowledge.web import router as knowledge_web_router
from .api_v2 import router as api_v2_router
from .api_reframe import router as api_reframe_router, web_router as reframe_web_router
from .api_rebuild import router as api_rebuild_router, web_router as rebuild_web_router
from .api_segment_fill import router as api_segment_fill_router
from .api_rebuild_run import router as api_rebuild_run_router
from .autopilot.api import router as autopilot_router, web_router as autopilot_web_router
from .csrf import CrossSiteRequestMiddleware
from .public import public_links_enabled, router as public_router
from .tasks import ExportEnqueueError
from .web_overlay import router as web_overlay_router
from .web_v2 import router as web_v2_router

app = FastAPI(title="re-skin", description="Video re-skinning tool")

# AICP-152: refuse cross-site POST/PUT/PATCH/DELETE (basic auth alone lets any
# page the operator visits drive the dashboard). Rule + curl policy: app/csrf.py.
app.add_middleware(CrossSiteRequestMiddleware)


@app.exception_handler(ExportEnqueueError)
async def _export_enqueue_error(request: Request, exc: ExportEnqueueError):
    """503 with a usable message when an export could not be queued at all.

    The endpoint has already rolled the optimistic stamp back, so the operator's
    Send button comes back enabled and the sheet state is honest. The export
    button JS renders ``detail``, so this reads as a real instruction instead of
    "Internal Server Error".
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# Static assets (CSS, htmx.min.js, …)
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# Browsers probe /favicon.ico on their own (raw JSON pages, media links) —
# serve the same icon the templates link to instead of a 404 per tab.
@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "favicon.ico"),
                        media_type="image/x-icon",
                        headers={"Cache-Control": "public, max-age=86400"})

# v2 REST API
app.include_router(api_v2_router, prefix="/api/v2")
app.include_router(api_assistant_router, prefix="/api/v2")
app.include_router(knowledge_router, prefix="/api/v2")  # 404 unless KNOWLEDGE_ENABLED
app.include_router(api_reframe_router, prefix="/api/v2")  # AICP-251
app.include_router(api_rebuild_router, prefix="/api/v2")  # rebuild wave 2a (REBUILD_ENABLED)
app.include_router(api_segment_fill_router, prefix="/api/v2")  # AICP-358
app.include_router(api_rebuild_run_router, prefix="/api/v2")  # rebuild images-only runs
app.include_router(autopilot_router, prefix="/api/v2")  # AICP-363
app.include_router(api_ops_router, prefix="/api/v2")  # AICP-350 drain status
app.include_router(api_paid_attempts_router, prefix="/api/v2")  # AICP-351 parked attempts

# Overlay feature REST API (standalone PNG-burn)
app.include_router(api_overlay_router, prefix="/api/v2/overlays")

# Public, token-signed media access (nginx serves /public without basic auth)
app.include_router(public_router, prefix="/public")

# AICP-151: say so loudly at boot (the app still starts; /public/* answers 503).
public_links_enabled()

# v2 web UI — mounted at /v2 (the primary operator UI)
app.include_router(web_v2_router, prefix="/v2")
# Run-page Outcome card (AICP-299) — 404 unless KNOWLEDGE_ENABLED + KNOWLEDGE_FEEDBACK_UI
app.include_router(knowledge_web_router, prefix="/v2")
app.include_router(reframe_web_router, prefix="/v2")
app.include_router(rebuild_web_router, prefix="/v2")
app.include_router(autopilot_web_router, prefix="/v2")  # AICP-363 card
app.include_router(ops_web_router, prefix="/v2")  # AICP-350 deploy banner

# Overlay web UI — mounted at /v2/overlays
app.include_router(web_overlay_router, prefix="/v2/overlays")


@app.get("/")
def root_redirect() -> RedirectResponse:
    """The old v1 dashboard lived at /; v2 is now the home. Redirect there."""
    return RedirectResponse(url="/v2/", status_code=307)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
