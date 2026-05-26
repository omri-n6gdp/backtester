"""
Phase 17.1 — API tests for POST /backtest/multi_ticker.
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


def _payload(strategy_id, tickers, **overrides):
    payload = {
        "strategy_id": strategy_id,
        "tickers": tickers,
        "timeframe": "1d",
        "start": "2010-01-01",
        "end": "2012-01-01",
        "params": {"short": 5, "long": 20},
    }
    payload.update(overrides)
    return payload


def test_multi_ticker_runs_each_ticker(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)

    resp = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY", "QQQ", "GLD"]),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {r["ticker"] for r in body["results"]} == {"SPY", "QQQ", "GLD"}
    assert body["aggregate"]["n_succeeded"] == 3
    assert body["aggregate"]["n_failed"] == 0
    # Each successful row has an equity curve.
    for r in body["results"]:
        assert r["error"] is None
        assert len(r["equity"]) > 0
        assert len(r["equity"]) == len(r["timestamps"])
        assert r["summary"]["final_equity"] is not None


def test_multi_ticker_isolates_per_ticker_errors(api_client, monkeypatch):
    """A ticker whose fetch fails surfaces as an error row; the rest of
    the batch is unaffected."""
    def get_prices_router(ticker, timeframe, start, end):
        if ticker == "BROKEN":
            raise RuntimeError("price provider exploded")
        return make_synthetic_ohlcv(n=500)

    install_fake_prices(monkeypatch, get_prices_router)
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)

    body = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY", "BROKEN", "QQQ"]),
    ).json()
    by_ticker = {r["ticker"]: r for r in body["results"]}
    assert by_ticker["SPY"]["error"] is None
    assert by_ticker["QQQ"]["error"] is None
    assert by_ticker["BROKEN"]["error"] is not None
    assert "exploded" in by_ticker["BROKEN"]["error"]
    assert body["aggregate"]["n_succeeded"] == 2
    assert body["aggregate"]["n_failed"] == 1


def test_multi_ticker_deduplicates_input(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)

    body = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY", "spy", "  SPY  ", "QQQ"]),
    ).json()
    tickers = [r["ticker"] for r in body["results"]]
    assert tickers == ["SPY", "QQQ"]


def test_multi_ticker_aggregate_picks_best_and_worst(api_client, monkeypatch):
    """The aggregate identifies the highest- and lowest-CAGR successful
    tickers."""
    # Two distinct synthetic series so CAGRs differ deterministically.
    def get_prices_router(ticker, timeframe, start, end):
        # The seed feeds the noise; differ by ticker so SPY > QQQ in
        # CAGR consistently for SMA(5/20).
        return make_synthetic_ohlcv(n=500, seed=1 if ticker == "SPY" else 2)

    install_fake_prices(monkeypatch, get_prices_router)
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)

    body = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY", "QQQ"]),
    ).json()
    agg = body["aggregate"]
    assert agg["best_ticker"] in {"SPY", "QQQ"}
    assert agg["worst_ticker"] in {"SPY", "QQQ"}
    assert agg["best_ticker"] != agg["worst_ticker"]
    # Mean/median are populated when at least one ticker succeeded.
    assert agg["mean_cagr"] is not None
    assert agg["median_cagr"] is not None


def test_multi_ticker_unknown_strategy_returns_404(api_client):
    resp = api_client.post(
        "/backtest/multi_ticker",
        json=_payload("nope", ["SPY"]),
    )
    assert resp.status_code == 404


def test_multi_ticker_rejects_empty_ticker_list(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)
    resp = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, []),
    )
    # Pydantic min_length=1 → 422.
    assert resp.status_code == 422
