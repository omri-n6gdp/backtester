"""
Shared fixtures and helpers for the API test suite.

Phase 5.9 consolidates the boilerplate that lived in four near-identical
copies across test_api.py, test_api_oos.py, test_api_walk_forward.py,
test_sweep.py, and (new) test_api_robustness.py.

Helpers exposed here are imported by test files via
    from tests.conftest import make_synthetic_ohlcv, post_strategy, install_fake_prices

Fixtures (api_client) are picked up automatically by pytest discovery.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def make_synthetic_ohlcv(
    n: int = 500,
    seed: int = 42,
    start: str = "2010-01-01",
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV. Business days from `start`."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 10_000_000, size=n).astype(float)
    idx = pd.date_range(start=start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    ).rename_axis("timestamp")


def install_fake_prices(monkeypatch: pytest.MonkeyPatch, fn: Callable) -> None:
    """Patch get_prices in every module that imports it by name.

    backend.api.routes / backend.deploy.scheduler / backend.engine.benchmarks
    all do `from backend.data.prices import get_prices` at module load
    and so keep their own reference. Patching only the data module is
    not enough — any of those modules would still see the original.
    """
    monkeypatch.setattr("backend.data.prices.get_prices", fn)
    monkeypatch.setattr("backend.api.routes.get_prices", fn)
    monkeypatch.setattr("backend.engine.benchmarks.get_prices", fn)


def post_strategy(client: TestClient, name: str, code: str, description: str = "") -> str:
    """POST a strategy through /strategies, asserting success. Returns the id."""
    resp = client.post(
        "/strategies",
        json={"name": name, "description": description, "code": code},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["strategy_id"]


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Fresh FastAPI app + isolated tmp storage + migrated DB.

    Does NOT patch get_prices — tests should install their own via
    `install_fake_prices(monkeypatch, fn)`. This keeps the fixture
    minimal and avoids accidental cross-test coupling on price shapes.
    """
    data_root = tmp_path / "btdata"
    db_dir = data_root / "results"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "results.db"
    strategies_dir = data_root / "strategies"
    strategies_dir.mkdir(parents=True)

    monkeypatch.setattr("backend.storage.DATA_ROOT", data_root, raising=False)
    monkeypatch.setattr("backend.storage.DB_DIR", db_dir, raising=False)
    monkeypatch.setattr("backend.storage.DB_PATH", db_path, raising=False)
    monkeypatch.setattr("backend.storage.STRATEGIES_DIR", strategies_dir, raising=False)
    monkeypatch.setattr("backend.api.routes.STRATEGIES_DIR", strategies_dir, raising=False)

    from backend.db.conn import migrate
    migrate()
    from backend.api import create_app
    from backend.deploy import scheduler

    # Each API test gets a clean scheduler singleton: the deployment
    # routes mutate it (registering cron jobs), and it must not leak
    # between tests.
    scheduler._reset_scheduler()
    app = create_app()
    with TestClient(app) as client:
        yield client
    scheduler._reset_scheduler()
