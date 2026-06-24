"""
FastAPI application for re-skin.
Wires together the REST API (/api) and the operator web UI (/).
"""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router as api_router
from .api_v2 import router as api_v2_router
from .public import router as public_router
from .web import router as web_router
from .web_v2 import router as web_v2_router

app = FastAPI(title="re-skin", description="Video re-skinning tool")

# Static assets (CSS, htmx.min.js, …)
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# REST API (JSON)
app.include_router(api_router, prefix="/api")

# v2 REST API
app.include_router(api_v2_router, prefix="/api/v2")

# Public, token-signed media access (nginx serves /public without basic auth)
app.include_router(public_router, prefix="/public")

# Operator web UI (HTML) — no prefix so it handles /  and  /jobs/{id}
app.include_router(web_router)

# v2 web UI — mounted at /v2
app.include_router(web_v2_router, prefix="/v2")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
