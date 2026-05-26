"""
FastAPI app factory.

Why a factory: tests instantiate a fresh app per session against a tmp DB,
and we want the startup work (schema migration) to fire on every
instantiation rather than on import.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

VERSION = "0.3.0"


def create_app(*, with_scheduler: bool = False) -> FastAPI:
    """Build the FastAPI app.

    `with_scheduler` is opt-in and only `server.py` passes it True. The
    deployment scheduler spins up a real background thread (and can place
    orders), so the test suite — which builds its own app via plain
    `create_app()` — must never start it.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Startup: schema migration runs on every boot. CREATE TABLE IF
        # NOT EXISTS keeps it idempotent; safe to run on a populated DB.
        from backend.storage import ensure_dirs
        from backend.db.conn import migrate, seed_cookbook_strategies

        ensure_dirs()
        migrate()
        # Phase 16.5 — seed worked-example strategies on first boot.
        # Skipped when running under the test fixture (which calls
        # `create_app()` without scheduler AND doesn't trigger lifespan
        # for synchronous TestClients in the same way). Idempotent: a
        # second startup with the same DB inserts nothing.
        if with_scheduler:
            seed_cookbook_strategies()

        if with_scheduler:
            from backend.deploy import scheduler
            scheduler.start()
        try:
            yield
        finally:
            # Shutdown: stop the scheduler thread cleanly.
            if with_scheduler:
                from backend.deploy import scheduler
                scheduler.shutdown()

    app = FastAPI(
        title="Backtester API",
        version=VERSION,
        description="Local-only backtesting engine. Bound to 127.0.0.1.",
        lifespan=lifespan,
    )

    # CORS (Phase 12.8 hardening). The frontend is served by this same
    # server, so app->API calls are same-origin and don't need CORS at
    # all. We list the two local origins explicitly rather than "*" so a
    # malicious page open in the same browser cannot POST to the
    # order-placing routes (/deployments, /killswitch).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from backend.api.routes import router

    app.include_router(router)
    return app
