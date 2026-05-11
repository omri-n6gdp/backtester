"""
Unit tests for backend/engine/robustness/regimes.py.

All tests use synthetic SPY/VIX series — no network. The actual SPY/VIX
fetching lives in the route layer and is covered by manual smoke testing.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.engine.robustness.regimes import (
    RegimeConfig,
    RegimeAnalysisResult,
    RegimeAxisResult,
    RegimeWindow,
    analyze_regimes,
    compute_spy_trend_regime,
    compute_vix_regime,
    partition_by_regime,
)
from backend.engine.vectorized import BacktestResult, Trade


# --- Fixtures ---------------------------------------------------------------

def _make_result(
    n_bars: int,
    daily_return: float = 0.0005,
    starting_capital: float = 100_000.0,
    seed: int = 0,
) -> BacktestResult:
    """Build a synthetic BacktestResult with deterministic per-bar returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B", tz="UTC")
    rets = pd.Series(rng.normal(daily_return, 0.01, n_bars), index=idx)
    rets.iloc[0] = 0.0
    equity = (1.0 + rets).cumprod() * starting_capital
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    positions = pd.Series(np.where(rng.random(n_bars) > 0.5, 1.0, 0.0), index=idx)
    raw_positions = positions.copy()
    costs = pd.Series(np.zeros(n_bars), index=idx)
    return BacktestResult(
        equity=equity, drawdown=drawdown, returns=rets,
        positions=positions, raw_positions=raw_positions, costs=costs,
        trades=[], starting_capital=starting_capital, bars_per_year=252,
    )


def _make_trade(entry_idx, exit_idx, pnl_return=0.05, side="long"):
    return Trade(
        entry_time=entry_idx,
        exit_time=exit_idx,
        side=side,
        avg_position=1.0,
        entry_price=100.0,
        exit_price=100.0 * (1 + pnl_return),
        bars_held=10,
        pnl_return=pnl_return,
    )


# --- compute_spy_trend_regime ----------------------------------------------

def test_spy_trend_uniformly_rising_is_all_bull() -> None:
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    spy = pd.Series(np.linspace(100, 200, n), index=idx)
    regime = compute_spy_trend_regime(spy, window=200)
    post_warmup = regime.iloc[200:].dropna()
    assert (post_warmup == "bull").all()


def test_spy_trend_uniformly_falling_is_all_bear() -> None:
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    spy = pd.Series(np.linspace(200, 100, n), index=idx)
    regime = compute_spy_trend_regime(spy, window=200)
    post_warmup = regime.iloc[200:].dropna()
    assert (post_warmup == "bear").all()


def test_spy_trend_warmup_bars_are_nan() -> None:
    n = 250
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    spy = pd.Series(np.linspace(100, 200, n), index=idx)
    regime = compute_spy_trend_regime(spy, window=200)
    assert regime.iloc[:199].isna().all()
    assert regime.iloc[200:].notna().all()


def test_spy_trend_crossover_flips_label() -> None:
    """A series that rises, then crashes below MA, then recovers should
    flip labels accordingly."""
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    arr = np.concatenate([
        np.linspace(100, 200, 250),
        np.full(50, 200),
        np.linspace(200, 50, 200),
        np.linspace(50, 200, 100),
    ])
    spy = pd.Series(arr, index=idx)
    regime = compute_spy_trend_regime(spy, window=200)
    bear_zone = regime.iloc[450:500].dropna()
    assert (bear_zone == "bear").any()
    bull_zone = regime.iloc[-50:].dropna()
    assert (bull_zone == "bull").any()


def test_spy_trend_window_validation() -> None:
    spy = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="window"):
        compute_spy_trend_regime(spy, window=1)


# --- compute_vix_regime ----------------------------------------------------

def test_vix_three_regimes_partition_correctly() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    vix = pd.Series([10.0, 14.99, 15.0, 25.0, 25.01], index=idx)
    regime = compute_vix_regime(vix)
    assert regime.iloc[0] == "low_vol"
    assert regime.iloc[1] == "low_vol"
    assert regime.iloc[2] == "normal_vol"
    assert regime.iloc[3] == "normal_vol"
    assert regime.iloc[4] == "high_vol"


def test_vix_custom_thresholds() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    vix = pd.Series([19.0, 20.0, 21.0], index=idx)
    regime = compute_vix_regime(vix, low_threshold=19.5, high_threshold=20.5)
    assert regime.iloc[0] == "low_vol"
    assert regime.iloc[1] == "normal_vol"
    assert regime.iloc[2] == "high_vol"


def test_vix_threshold_validation() -> None:
    vix = pd.Series([15.0, 20.0, 25.0])
    with pytest.raises(ValueError, match="low_threshold"):
        compute_vix_regime(vix, low_threshold=25.0, high_threshold=15.0)


def test_vix_nan_passes_through() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    vix = pd.Series([10.0, float("nan"), 30.0], index=idx)
    regime = compute_vix_regime(vix)
    assert regime.iloc[0] == "low_vol"
    assert pd.isna(regime.iloc[1])
    assert regime.iloc[2] == "high_vol"


# --- partition_by_regime ----------------------------------------------------

def test_partition_single_regime_matches_full_metrics() -> None:
    """If every bar has the same regime label, that one window's
    bar-level metrics must match the full-result metrics exactly."""
    result = _make_result(n_bars=300, seed=1)
    label_series = pd.Series("only", index=result.returns.index, dtype=object)

    axis = partition_by_regime(result, label_series, axis_name="trivial")
    assert axis.n_bars_total == 300
    assert axis.n_bars_unknown == 0
    assert set(axis.windows.keys()) == {"only"}
    w = axis.windows["only"]
    assert w.n_bars == 300
    assert math.isclose(w.fraction, 1.0)

    expected_total_return = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1.0)
    assert math.isclose(w.metrics.total_return, expected_total_return, rel_tol=1e-10)


def test_partition_two_regimes_split_correctly() -> None:
    result = _make_result(n_bars=100, seed=2)
    labels = ["a"] * 60 + ["b"] * 40
    label_series = pd.Series(labels, index=result.returns.index, dtype=object)

    axis = partition_by_regime(result, label_series, axis_name="ab")
    assert set(axis.windows.keys()) == {"a", "b"}
    assert axis.windows["a"].n_bars == 60
    assert axis.windows["b"].n_bars == 40
    assert math.isclose(axis.windows["a"].fraction, 0.6)
    assert math.isclose(axis.windows["b"].fraction, 0.4)
    assert axis.n_bars_unknown == 0


def test_partition_unknown_bars_excluded_and_counted() -> None:
    result = _make_result(n_bars=100, seed=3)
    labels = [np.nan] * 30 + ["x"] * 70
    label_series = pd.Series(labels, index=result.returns.index, dtype=object)

    axis = partition_by_regime(result, label_series, axis_name="warmup")
    assert axis.n_bars_unknown == 30
    assert "x" in axis.windows
    assert axis.windows["x"].n_bars == 70
    assert math.isclose(axis.windows["x"].fraction, 1.0)


def test_partition_per_regime_equity_rebased_to_starting_capital() -> None:
    result = _make_result(n_bars=100, daily_return=0.001, seed=4)
    labels = ["g"] * 100
    label_series = pd.Series(labels, index=result.returns.index, dtype=object)
    axis = partition_by_regime(result, label_series, axis_name="trivial")
    w = axis.windows["g"]
    assert w.metrics.total_return > 0


def test_partition_trades_attributed_by_entry_regime() -> None:
    result = _make_result(n_bars=200, seed=5)
    idx = result.returns.index
    labels = ["early"] * 100 + ["late"] * 100
    label_series = pd.Series(labels, index=idx, dtype=object)

    result.trades.append(_make_trade(idx[10], idx[50], pnl_return=0.05))
    result.trades.append(_make_trade(idx[120], idx[140], pnl_return=-0.02))
    result.trades.append(_make_trade(idx[150], idx[180], pnl_return=0.03))

    axis = partition_by_regime(result, label_series, axis_name="time")
    assert axis.windows["early"].metrics.n_trades == 1
    assert axis.windows["late"].metrics.n_trades == 2


def test_partition_handles_lower_freq_regime_via_ffill() -> None:
    """A daily regime series should label a higher-freq backtest correctly
    via forward-fill."""
    bt_idx = pd.date_range("2020-01-06 09:30", periods=100, freq="h", tz="UTC")
    rets = pd.Series(0.001, index=bt_idx)
    rets.iloc[0] = 0.0
    equity = (1.0 + rets).cumprod() * 100_000
    drawdown = equity / equity.cummax() - 1.0
    result = BacktestResult(
        equity=equity, drawdown=drawdown, returns=rets,
        positions=pd.Series(1.0, index=bt_idx),
        raw_positions=pd.Series(1.0, index=bt_idx),
        costs=pd.Series(0.0, index=bt_idx),
        trades=[], starting_capital=100_000.0, bars_per_year=252 * 24,
    )

    daily_idx = pd.date_range("2020-01-06", periods=6, freq="D", tz="UTC")
    daily_labels = pd.Series(["a", "b", "a", "b", "a", "b"], index=daily_idx, dtype=object)

    axis = partition_by_regime(result, daily_labels, axis_name="daily_axis")
    assert axis.n_bars_unknown == 0
    assert set(axis.windows.keys()) == {"a", "b"}


def test_partition_skips_singleton_regime() -> None:
    """A regime with fewer than 2 bars can't have meaningful metrics —
    it should be silently dropped from the windows dict."""
    result = _make_result(n_bars=50, seed=6)
    labels = ["main"] * 49 + ["rare"]
    label_series = pd.Series(labels, index=result.returns.index, dtype=object)

    axis = partition_by_regime(result, label_series, axis_name="ax")
    assert "main" in axis.windows
    assert "rare" not in axis.windows


# --- analyze_regimes orchestrator ------------------------------------------

def test_analyze_regimes_both_axes_populated() -> None:
    result = _make_result(n_bars=400, seed=7)
    spy_idx = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    spy = pd.Series(np.linspace(100, 200, 400), index=spy_idx)
    vix = pd.Series(np.linspace(10, 30, 400), index=spy_idx)

    out = analyze_regimes(result, spy_close=spy, vix_close=vix)
    assert out.spy_trend is not None
    assert out.vix_level is not None
    assert out.spy_trend_error is None
    assert out.vix_level_error is None
    assert "bull" in out.spy_trend.windows
    assert {"low_vol", "normal_vol", "high_vol"} <= set(out.vix_level.windows.keys())


def test_analyze_regimes_missing_spy_only_gives_vix() -> None:
    result = _make_result(n_bars=200, seed=8)
    vix_idx = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    vix = pd.Series(20.0, index=vix_idx)
    out = analyze_regimes(result, spy_close=None, vix_close=vix)
    assert out.spy_trend is None
    assert out.spy_trend_error == "SPY data not provided"
    assert out.vix_level is not None


def test_analyze_regimes_missing_vix_only_gives_spy() -> None:
    result = _make_result(n_bars=400, seed=9)
    spy_idx = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    spy = pd.Series(np.linspace(100, 200, 400), index=spy_idx)
    out = analyze_regimes(result, spy_close=spy, vix_close=None)
    assert out.spy_trend is not None
    assert out.vix_level is None
    assert out.vix_level_error == "VIX data not provided"


def test_analyze_regimes_handles_empty_spy_series() -> None:
    result = _make_result(n_bars=100, seed=10)
    empty = pd.Series([], dtype=float)
    out = analyze_regimes(result, spy_close=empty, vix_close=empty)
    assert out.spy_trend is None
    assert out.vix_level is None


def test_analyze_regimes_custom_config() -> None:
    result = _make_result(n_bars=300, seed=11)
    spy_idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    spy = pd.Series(np.linspace(100, 200, 300), index=spy_idx)
    cfg = RegimeConfig(spy_trend_window=50)
    out = analyze_regimes(result, spy_close=spy, vix_close=None, config=cfg)
    assert out.spy_trend.n_bars_unknown < 200
