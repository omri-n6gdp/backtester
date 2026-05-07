"""
Entry point: `python server.py` boots the API on http://127.0.0.1:8000.

To enable autoreload during development, run instead:
    uvicorn server:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import uvicorn

from backend.api import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
