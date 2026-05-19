"""Trading 212 deployment pipeline.

Phase 11 components (built foundation-first):
  - runner.py:       compute target positions, diff against T212 portfolio,
                     produce OrderIntent deltas (pure compute).
  - orders.py:       persist + submit intents idempotently via the T212 client.
  - scheduler.py:    APScheduler hook (Phase 11.5).

All Demo-only in Phase 11; live trading is gated to Phase 13.
"""
