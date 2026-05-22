"""
Entry point. Runs FastAPI on 127.0.0.1:8000 AND serves frontend/ as static
files at the same origin, so the browser can load app.js as an ES module
without hitting CORS / file:// origin issues.
"""
from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

from backend.api import create_app
from backend.logging_config import setup_logging

# Wire the root logger to ~/.backtester/logs/ + console before anything
# else, so app construction and the lifespan startup are captured too.
setup_logging()

# --- PHASE4_FRONTEND_MOUNT ---
# with_scheduler=True: the deployment scheduler starts/stops with the
# server via the app's lifespan handler. The test suite builds its own
# app via plain create_app() and so never spins up the scheduler thread.
app = create_app(with_scheduler=True)

# Serve frontend/ at the root. html=True makes /  serve index.html.
# Mounted last so API routes (registered inside create_app) take precedence.
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
