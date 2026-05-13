"""
API-level tests for the robustness features bolted onto POST /backtest:
monte_carlo, n_trials (DSR), and include_regimes. Engine-level math is
covered in test_monte_carlo.py / test_deflated_sharpe.py / test_regimes.py;
this file verifies route wiring (Pydantic marshalling, error handling,
response shape).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import install_fake_prices, make_synthetic_ohlcv, post_strategy


# A strategy that produces multiple round-trip trades (needed for Monte Carlo
# to have something to shuffle). With noisy synthetic data, SMA(5/20) will
# toggle several times.
SMA_STRATEGY = """
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 5))
    long = int(params.get('long', 20))
    sma_s = data['close'].rolling(short).mean()
    sma_l = data['close'].rolling(long).mean()
    return (sma_s > sma_l).astype(float)
"""


def _make_vix_close(n: int = 500, start: str = "2010-01-01") -> pd.Series:
    """Synthetic VIX series oscillating across the 10-30 range so both
    low_vol (<15) and normal_vol (15-25) regime buckets get populated."""
    rng = np.random.default_rng(99)
    base = 20 + 8 * np.sin(np.linspace(0, 4 * np.pi, n))
    vix = base + rng.normal(0, 1.5, n)
    idx = pd.date_range(start=start, periods=n, freq="B", tz="UTC")
    return pd.Series(vix, index=idx)


@pytest.fixture
def client_with_strategy(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA 5/20", SMA_STRATEGY)
    return api_client, sid


def _payload(strategy_id, **overrides):
    payload = {
        "strategy_id": strategy_id,
        "ticker": "SPY",
        "timeframe": "1d",
        "start": "2010-01-01",
        "end": "2012-01-01",
        "params": {"short": 5, "long": 20},
    }
    payload.update(overrides)
    return payload


# ---- baseline: optional fields default to None --------------------------

def test_backtest_without_robustness_omits_optional_fields(client_with_strategy):
    client, sid = client_with_strategy
    resp = client.post("/backtest", json=_payload(sid))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["oos"] is None
    assert body["monte_carlo"] is None
    assert body["deflated_sharpe"] is None
    assert body["regimes"] is None


# ---- Monte Carlo --------------------------------------------------------

def test_backtest_with_monte_carlo_returns_band_data(client_with_strategy):
    client, sid = client_with_strategy
    payload = _payload(sid, monte_carlo={"n_iterations": 100, "seed": 42})
    resp = client.post("/backtest", json=payload)
    assert resp.status_code == 200, resp.text
    mc = resp.json()["monte_carlo"]
    assert mc is not None
    assert mc["n_iterations"] == 100
    assert mc["seed"] == 42
    assert mc["n_trades"] >= 2, "SMA crossover should produce >= 2 trades on noisy data"
    assert "equity_bands" in mc and isinstance(mc["equity_bands"], dict)
    assert "max_dd_quantiles" in mc and len(mc["max_dd_quantiles"]) >= 1


def test_backtest_with_invalid_mc_iterations_returns_400(client_with_strategy):
    client, sid = client_with_strategy
    resp = client.post("/backtest", json=_payload(sid, monte_carlo={"n_iterations": 0}))
    assert resp.status_code == 400, resp.text


# ---- Deflated Sharpe (n_trials) ----------------------------------------

def test_backtest_with_n_trials_returns_deflated_sharpe(client_with_strategy):
    client, sid = client_with_strategy
    resp = client.post("/backtest", json=_payload(sid, n_trials=9))
    assert resp.status_code == 200, resp.text
    ds = resp.json()["deflated_sharpe"]
    assert ds is not None
    assert ds["n_trials"] == 9
    assert "psr" in ds
    assert "dsr" in ds
    assert "sharpe_annualized" in ds


def test_backtest_with_n_trials_zero_returns_400(client_with_strategy):
    client, sid = client_with_strategy
    resp = client.post("/backtest", json=_payload(sid, n_trials=0))
    assert resp.status_code == 400, resp.text


# ---- Regime analysis ---------------------------------------------------

def test_backtest_with_include_regimes_returns_both_axes(api_client, monkeypatch):
    """SPY and ^VIX are fetched by the route. We stub get_prices so the
    SMA strategy gets one shape and the regime helper gets SPY/VIX-shaped
    data with enough variation that both axes populate at least one window.
    """
    def get_prices_router(ticker, timeframe, start, end):
        n = 500
        if ticker == "^VIX":
            close = _make_vix_close(n).to_numpy()
        else:
            # SPY (or anything else) gets the standard synthetic OHLCV.
            return make_synthetic_ohlcv(n=n)
        idx = pd.date_range("2010-01-01", periods=n, freq="B", tz="UTC")
        return pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e6},
            index=idx,
        )

    install_fake_prices(monkeypatch, get_prices_router)
    sid = post_strategy(api_client, "SMA 5/20", SMA_STRATEGY)

    resp = api_client.post("/backtest", json=_payload(sid, include_regimes=True))
    assert resp.status_code == 200, resp.text
    regimes = resp.json()["regimes"]
    assert regimes is not None
    # Either axis populated or surfaced as an error — both are valid responses.
    assert ("spy_trend" in regimes) or regimes.get("spy_trend_error")
    assert ("vix_level" in regimes) or regimes.get("vix_level_error")
    if regimes.get("vix_level"):
        # VIX synthesis spans 10-30, so at least one regime bucket should
        # be non-empty.
        windows = regimes["vix_level"]["windows"]
        non_empty = [w for w in windows.values() if w["n_bars"] > 0]
        assert len(non_empty) >= 1
