"""
Phase 16.5 — unit tests for `seed_cookbook`.

Uses the real cookbook manifest and source files (under `seeds/cookbook/`);
the seeder writes into the tmp_path-redirected STRATEGIES_DIR and DB.
"""
from __future__ import annotations

from backend.db.conn import connect, migrate, seed_cookbook_strategies
from backend.seeds.cookbook import seed_cookbook


def test_seed_cookbook_inserts_six_strategies(api_client):
    """Fresh DB → six strategies inserted, each with its own initial
    version row."""
    # `api_client` fixture has already called migrate(); the DB has
    # no strategies. We seed by calling the public top-level helper.
    inserted = seed_cookbook_strategies()
    assert inserted == 6

    rows = api_client.get("/strategies").json()
    names = sorted(r["name"] for r in rows)
    assert names == sorted([
        "Cookbook · SMA crossover",
        "Cookbook · RSI mean reversion",
        "Cookbook · MACD signal",
        "Cookbook · 12-1 momentum",
        "Cookbook · Bollinger breakout",
        "Cookbook · Donchian channel",
    ])

    # Each cookbook strategy has exactly one version row.
    for r in rows:
        versions = api_client.get(f"/strategies/{r['id']}/versions").json()
        assert len(versions["versions"]) == 1


def test_seed_cookbook_is_idempotent(api_client):
    """Running the seeder twice doesn't duplicate rows."""
    first = seed_cookbook_strategies()
    second = seed_cookbook_strategies()
    assert first == 6
    assert second == 0
    rows = api_client.get("/strategies").json()
    assert len(rows) == 6


def test_seed_cookbook_validates_each_strategy(api_client):
    """Every seeded strategy passes /strategies/validate (proof the
    validator approves of all cookbook code)."""
    seed_cookbook_strategies()
    for r in api_client.get("/strategies").json():
        detail = api_client.get(f"/strategies/{r['id']}").json()
        v = api_client.post(
            "/strategies/validate", json={"code": detail["code"]},
        ).json()
        assert v["ok"], f"{r['name']} failed: {v['errors']}"
