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


# ---- Phase 18.1 — per-asset cost overrides ----------------------------


def test_per_asset_override_visibly_drags_final_equity(api_client, monkeypatch):
    """A heavy slippage override on one ticker drags its final equity
    vs the same ticker without overrides; other tickers stay tight."""
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)

    # No overrides — baseline.
    base = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY", "PENNY"]),
    ).json()
    by_t = {r["ticker"]: r for r in base["results"]}
    baseline_penny = by_t["PENNY"]["summary"]["final_equity"]
    baseline_spy = by_t["SPY"]["summary"]["final_equity"]

    # Override PENNY with much higher slippage; SPY untouched.
    with_overrides = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY", "PENNY"], cost_overrides={
            "PENNY": {"slippage_bps": 200.0},
        }),
    ).json()
    by_t2 = {r["ticker"]: r for r in with_overrides["results"]}
    overridden_penny = by_t2["PENNY"]["summary"]["final_equity"]
    overridden_spy = by_t2["SPY"]["summary"]["final_equity"]

    # PENNY paid more in slippage; SPY is unchanged.
    assert overridden_penny < baseline_penny
    assert abs(overridden_spy - baseline_spy) < 1e-6


def test_override_unknown_field_rejected(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)
    resp = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY"], cost_overrides={
            "SPY": {"not_a_field": 99},
        }),
    )
    assert resp.status_code == 422


def test_override_unknown_ticker_silently_ignored(api_client, monkeypatch):
    """An override for a ticker that isn't in the list should not error
    — the user might paste a stale override map."""
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)
    resp = api_client.post(
        "/backtest/multi_ticker",
        json=_payload(sid, ["SPY"], cost_overrides={
            "OLD": {"slippage_bps": 100.0},
        }),
    )
    assert resp.status_code == 200
    assert resp.json()["aggregate"]["n_succeeded"] == 1
