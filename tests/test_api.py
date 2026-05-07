"""
API tests. Each test gets a fresh app + isolated tmp DB + tmp strategies dir.

Why migrate() is called inside the fixture (and not via FastAPI startup):
the startup hook fires when the app is created, but our monkeypatch of
DB_PATH happens before that — yet the hook reads DB_PATH lazily inside
backend.db.conn.connect, so we'd be fine *except* that TestClient doesn't
reliably trigger startup hooks across all FastAPI/Starlette versions.
Calling migrate() explicitly is robust and faster.

get_prices is monkeypatched to a synthetic OHLCV frame so tests run offline.
"""
from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


SAMPLE_STRATEGY = textwrap.dedent(
    """
    import pandas as pd

    def strategy(data, params):
        short = int(params.get('short', 5))
        long = int(params.get('long', 20))
        sma_s = data['close'].rolling(short).mean()
        sma_l = data['close'].rolling(long).mean()
        return (sma_s > sma_l).astype(float)
    """
).strip()

INVALID_STRATEGY = "import os\n\ndef strategy(data, params):\n    return data['close'] * 0\n"


def _synthetic_ohlcv(n_bars: int = 252) -> pd.DataFrame:
    """Deterministic synthetic OHLCV — enough bars to satisfy 5/20 SMA crossover."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.012, size=n_bars)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
    volume = rng.integers(1_000_000, 10_000_000, size=n_bars).astype(float)
    idx = pd.date_range(end=datetime.now(timezone.utc).date(), periods=n_bars, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    ).rename_axis("timestamp")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh app + isolated storage for each test."""
    db_dir = tmp_path / "results"
    strategies_dir = tmp_path / "strategies"
    db_path = db_dir / "results.db"

    # Redirect storage paths BEFORE creating the app or running migrations.
    monkeypatch.setattr("backend.storage.DB_DIR", db_dir)
    monkeypatch.setattr("backend.storage.DB_PATH", db_path)
    monkeypatch.setattr("backend.storage.STRATEGIES_DIR", strategies_dir)
    # routes.py imports STRATEGIES_DIR by name at module load, so patch there too:
    monkeypatch.setattr("backend.api.routes.STRATEGIES_DIR", strategies_dir)

    # Stub get_prices in BOTH the data module and the routes module
    # (routes imports it by name at module load).
    def fake_get_prices(ticker, timeframe, start, end):
        if ticker == "BADTICKER":
            return pd.DataFrame()
        return _synthetic_ohlcv(252)

    monkeypatch.setattr("backend.data.prices.get_prices", fake_get_prices)
    monkeypatch.setattr("backend.api.routes.get_prices", fake_get_prices)

    # Create tables in the redirected DB. This is the key fix vs the
    # original test file: don't rely on the FastAPI startup hook firing
    # in TestClient — invoke the migration explicitly.
    from backend.db.conn import migrate
    migrate()

    from backend.api import create_app
    app = create_app()
    return TestClient(app)


# -------- /health --------

def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]


# -------- /prices --------

def test_prices_happy(client: TestClient) -> None:
    r = client.get("/prices", params={
        "ticker": "SPY", "timeframe": "1d",
        "start": "2024-01-01", "end": "2024-12-31",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "SPY"
    assert len(body["timestamps"]) == len(body["close"]) == 252
    assert all(c > 0 for c in body["close"])


def test_prices_empty_returns_404(client: TestClient) -> None:
    r = client.get("/prices", params={
        "ticker": "BADTICKER", "timeframe": "1d",
        "start": "2024-01-01", "end": "2024-12-31",
    })
    assert r.status_code == 404


# -------- /strategies --------

def test_create_strategy_happy(client: TestClient) -> None:
    r = client.post("/strategies", json={
        "name": "Test SMA",
        "description": "Test strategy",
        "code": SAMPLE_STRATEGY,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "strategy_id" in body
    assert body["validation_result"]["ok"] is True


def test_create_strategy_rejects_invalid_code(client: TestClient) -> None:
    r = client.post("/strategies", json={
        "name": "Bad",
        "description": "",
        "code": INVALID_STRATEGY,  # `import os` is banned by validator.
    })
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["message"] == "strategy validation failed"
    assert detail["errors"], "expected at least one validation error"


def test_list_strategies(client: TestClient) -> None:
    assert client.get("/strategies").json() == []
    client.post("/strategies", json={
        "name": "S1", "description": "first", "code": SAMPLE_STRATEGY,
    })
    rows = client.get("/strategies").json()
    assert len(rows) == 1
    assert rows[0]["name"] == "S1"


def test_get_strategy(client: TestClient) -> None:
    create = client.post("/strategies", json={
        "name": "S1", "description": "first", "code": SAMPLE_STRATEGY,
    }).json()
    sid = create["strategy_id"]
    detail = client.get(f"/strategies/{sid}").json()
    assert detail["id"] == sid
    assert "def strategy" in detail["code"]


def test_get_strategy_not_found(client: TestClient) -> None:
    r = client.get("/strategies/does-not-exist")
    assert r.status_code == 404


def test_delete_strategy_soft_deletes(client: TestClient) -> None:
    create = client.post("/strategies", json={
        "name": "S1", "description": "first", "code": SAMPLE_STRATEGY,
    }).json()
    sid = create["strategy_id"]

    r = client.delete(f"/strategies/{sid}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    assert client.get("/strategies").json() == []
    assert client.get(f"/strategies/{sid}").status_code == 404


# -------- /backtest --------

def test_backtest_happy(client: TestClient) -> None:
    create = client.post("/strategies", json={
        "name": "SMA",
        "description": "",
        "code": SAMPLE_STRATEGY,
    }).json()
    sid = create["strategy_id"]

    r = client.post("/backtest", json={
        "strategy_id": sid,
        "ticker": "SPY",
        "timeframe": "1d",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "params": {"short": 5, "long": 20},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "run_id" in body
    assert "metrics" in body
    assert len(body["equity"]) == len(body["timestamps"]) == 252
    assert body["equity"][-1] is not None


def test_backtest_unknown_strategy_404(client: TestClient) -> None:
    r = client.post("/backtest", json={
        "strategy_id": "does-not-exist",
        "ticker": "SPY",
        "timeframe": "1d",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "params": {},
    })
    assert r.status_code == 404
