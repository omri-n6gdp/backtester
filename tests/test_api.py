"""API tests covering /health, /prices, /strategies CRUD, and /backtest happy paths."""
from __future__ import annotations

import textwrap

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import install_fake_prices, make_synthetic_ohlcv


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


@pytest.fixture
def client(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def fake(ticker, timeframe, start, end):
        if ticker == "BADTICKER":
            return pd.DataFrame()
        return make_synthetic_ohlcv(n=252, start="2010-01-04")
    install_fake_prices(monkeypatch, fake)
    return api_client


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
