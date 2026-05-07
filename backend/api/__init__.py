"""
FastAPI app factory.

Why a factory: tests instantiate a fresh app per session against a tmp DB,
and we want startup hooks (schema migration) to fire on every instantiation
rather than on import.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

VERSION = "0.3.0"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Backtester API",
        version=VERSION,
        description="Local-only backtesting engine. Bound to 127.0.0.1.",
    )

    # Permissive CORS: server binds to localhost only, so cross-origin
    # protection isn't doing useful work here. file:// origins send
    # `Origin: null`, which "*" covers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Schema migration runs on every startup. CREATE TABLE IF NOT EXISTS
    # keeps it idempotent; safe to run on a populated DB.
    @app.on_event("startup")
    def _startup() -> None:
        from backend.storage import ensure_dirs
        from backend.db.conn import migrate

        ensure_dirs()
        migrate()

    from backend.api.routes import router

    app.include_router(router)
    return app
