"""
FastAPI application for re-skin.
Minimal skeleton with health check and placeholder landing page.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="re-skin", description="Video re-skinning tool")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>re-skin</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .container {
                text-align: center;
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 {
                margin: 0 0 10px 0;
                color: #333;
            }
            p {
                color: #666;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>re-skin is running</h1>
            <p>Video re-skinning tool</p>
        </div>
    </body>
    </html>
    """
