"""
API tests for GET /library and GET /runs/{run_id} (Phase 8).
"""
from __future__ import annotations

import pytest

from tests.conftest import install_fake_prices, make_synthetic_ohlcv, post_strategy


SAMPLE_STRATEGY = """
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 5))
    long = int(params.get('long', 20))
    return (data['close'].rolling(short).mean() > data['close'].rolling(long).mean()).astype(float)
"""


@pytest.fixture
def library_client(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SAMPLE_STRATEGY)
    return api_client, sid


# ---- /library ----------------------------------------------------------

def test_library_empty_when_no_runs(api_client):
    resp = api_client.get("/library")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_library_lists_runs_after_backtest(library_client):
    client, sid = library_client
    # Run two backtests on different tickers so breadth_score can populate.
    for ticker in ["SPY", "AAPL"]:
        resp = client.post("/backtest", json={
            "strategy_id": sid, "ticker": ticker, "timeframe": "1d",
            "start": "2010-01-01", "end": "2012-01-01",
            "params": {"short": 5, "long": 20},
        })
        assert resp.status_code == 200, resp.text

    body = client.get("/library").json()
    assert len(body["runs"]) == 2
    tickers = sorted(r["ticker"] for r in body["runs"])
    assert tickers == ["AAPL", "SPY"]
    # All scoring components present and in [0, 1] (or null for dsr).
    for r in body["runs"]:
        for key in ("sharpe_score", "calmar_score", "consistency_score",
                    "breadth_score", "recency_score"):
            v = r[key]
            assert 0.0 <= v <= 1.0, f"{key}={v} out of range"
        assert r["dsr_score"] is None
        # Strategy was tested on 2 distinct tickers; breadth is between 0 and 1.
        assert 0.0 < r["breadth_score"] < 1.0


def test_library_ticker_filter(library_client):
    client, sid = library_client
    for ticker in ["SPY", "AAPL", "QQQ"]:
        client.post("/backtest", json={
            "strategy_id": sid, "ticker": ticker, "timeframe": "1d",
            "start": "2010-01-01", "end": "2012-01-01",
            "params": {"short": 5, "long": 20},
        })

    body = client.get("/library", params={"ticker": "SPY"}).json()
    assert len(body["runs"]) == 1
    assert body["runs"][0]["ticker"] == "SPY"


# ---- /runs/{run_id} ----------------------------------------------------

def test_get_run_roundtrips_a_backtest(library_client):
    client, sid = library_client
    resp = client.post("/backtest", json={
        "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2012-01-01",
        "params": {"short": 5, "long": 20},
    })
    run_id = resp.json()["run_id"]
    original = resp.json()

    got = client.get(f"/runs/{run_id}").json()
    assert got["run_id"] == run_id
    assert len(got["timestamps"]) == len(original["timestamps"])
    assert got["equity"][-1] == original["equity"][-1]
    assert got["summary"]["sharpe"] == original["summary"]["sharpe"]
    assert len(got["trades"]) == len(original["trades"])
    # The runs route doesn't replay optional analyses — they're null.
    assert got["oos"] is None
    assert got["monte_carlo"] is None
    assert got["deflated_sharpe"] is None
    assert got["regimes"] is None


def test_get_run_unknown_id_returns_404(api_client):
    resp = api_client.get("/runs/nonexistent")
    assert resp.status_code == 404
