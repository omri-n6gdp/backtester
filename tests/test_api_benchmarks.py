"""
Phase 17.2 — API tests for the benchmark suite.
"""
from __future__ import annotations

import pandas as pd

from tests.conftest import install_fake_prices, make_synthetic_ohlcv, post_strategy


SMA_STRATEGY = """
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 5))
    long = int(params.get('long', 20))
    return (data['close'].rolling(short).mean() > data['close'].rolling(long).mean()).astype(float)
"""


def test_benchmarks_catalog_route(api_client):
    body = api_client.get("/benchmarks").json()
    ids = {b["id"] for b in body["benchmarks"]}
    assert {"spy_bh", "qqq_bh", "gold", "sixty_forty", "mag7_eq"}.issubset(ids)
    # Each entry carries a human label + tickers list.
    for b in body["benchmarks"]:
        assert b["label"]
        assert isinstance(b["tickers"], list)


def test_backtest_with_benchmarks_returns_curves(api_client, monkeypatch):
    """A backtest with a benchmark request returns an overlay curve
    aligned to the strategy's equity index."""
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=300))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)

    resp = api_client.post("/backtest", json={
        "strategy_id": sid,
        "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2011-06-01",
        "params": {"short": 5, "long": 20},
        "benchmarks": ["spy_bh", "qqq_bh"],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["benchmarks"]) == 2
    by_id = {b["id"]: b for b in body["benchmarks"]}
    for bid in ("spy_bh", "qqq_bh"):
        bm = by_id[bid]
        assert bm["error"] is None
        assert len(bm["equity"]) == len(body["timestamps"])
        # Aligned bar-for-bar with the strategy equity series.
        assert bm["timestamps"] == body["timestamps"]


def test_benchmark_with_no_data_returns_error_entry(api_client, monkeypatch):
    """An unfetchable ticker doesn't fail the backtest — the curve
    comes back with `error` populated and empty arrays."""
    def get_prices_router(ticker, timeframe, start, end):
        if ticker in {"GLD"}:
            raise RuntimeError("GLD provider offline")
        return make_synthetic_ohlcv(n=300)

    install_fake_prices(monkeypatch, get_prices_router)
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)
    body = api_client.post("/backtest", json={
        "strategy_id": sid,
        "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2011-06-01",
        "params": {"short": 5, "long": 20},
        "benchmarks": ["spy_bh", "gold"],
    }).json()
    by_id = {b["id"]: b for b in body["benchmarks"]}
    assert by_id["spy_bh"]["error"] is None
    assert by_id["gold"]["error"] is not None
    # Strategy itself still succeeded.
    assert body["summary"]["final_equity"] is not None


def test_unknown_benchmark_id_surfaces_as_error(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=200))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)
    body = api_client.post("/backtest", json={
        "strategy_id": sid,
        "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2011-01-01",
        "params": {"short": 5, "long": 20},
        "benchmarks": ["not_a_benchmark"],
    }).json()
    assert body["benchmarks"][0]["id"] == "not_a_benchmark"
    assert body["benchmarks"][0]["error"]


def test_no_benchmarks_default_omits_field(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=200))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)
    body = api_client.post("/backtest", json={
        "strategy_id": sid,
        "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2011-01-01",
        "params": {"short": 5, "long": 20},
    }).json()
    assert body["benchmarks"] == []
