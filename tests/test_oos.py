"""Tests for IS/OOS split (backend.engine.robustness.oos)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.engine.robustness.oos import OOSResult, WindowResult, split_result
from backend.engine.vectorized import run_backtest


# --- fixtures -------------------------------------------------------------


def _make_data(n_bars: int = 1000, seed: int = 42, tz="UTC") -> pd.DataFrame:
    """Synthetic OHLCV with GBM-ish daily returns."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n_bars)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, n_bars))
    low = close * (1 - rng.uniform(0, 0.01, n_bars))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n_bars))
    volume = rng.uniform(1e5, 1e7, n_bars)
    idx = pd.date_range("2010-01-01", periods=n_bars, freq="B", tz=tz)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _buy_and_hold(data: pd.DataFrame, params: dict) -> pd.Series:
    return pd.Series(1.0, index=data.index)


def _flip_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    """Long for the first third, short for the second, flat for the rest."""
    n = len(data)
    s = pd.Series(0.0, index=data.index)
    s.iloc[: n // 3] = 1.0
    s.iloc[n // 3 : 2 * n // 3] = -1.0
    return s


# --- core behaviour -------------------------------------------------------


def test_split_with_fraction_produces_correct_window_sizes():
    data = _make_data(1000)
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split=0.7)
    assert oos.in_sample.n_bars == 700
    assert oos.out_of_sample.n_bars == 300
    assert oos.split_index == 700
    assert oos.in_sample.label == "in_sample"
    assert oos.out_of_sample.label == "out_of_sample"


def test_split_with_string_date():
    data = _make_data(1000)
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split="2012-01-01")
    assert oos.split_timestamp >= pd.Timestamp("2012-01-01", tz="UTC")
    assert oos.in_sample.n_bars + oos.out_of_sample.n_bars == 1000


def test_split_with_pd_timestamp():
    data = _make_data(1000)
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split=pd.Timestamp("2012-06-01", tz="UTC"))
    assert oos.in_sample.end < oos.split_timestamp
    assert oos.out_of_sample.start >= oos.split_timestamp


def test_split_with_naive_index():
    data = _make_data(500, tz=None)
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split=0.5)
    assert oos.in_sample.n_bars == 250
    assert oos.out_of_sample.n_bars == 250


def test_split_with_naive_split_on_tz_aware_index():
    """A naive split timestamp is auto-localized to the index's tz."""
    data = _make_data(500, tz="UTC")
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split=pd.Timestamp("2010-12-01"))
    assert oos.split_timestamp.tzinfo is not None
    assert oos.in_sample.n_bars > 0
    assert oos.out_of_sample.n_bars > 0


# --- metrics --------------------------------------------------------------


def test_both_windows_have_full_metrics():
    data = _make_data(500)
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split=0.6)
    assert isinstance(oos.in_sample.metrics.sharpe, float)
    assert isinstance(oos.out_of_sample.metrics.sharpe, float)
    assert oos.in_sample.metrics.n_trades >= 0
    assert oos.out_of_sample.metrics.n_trades >= 0


def test_oos_equity_is_rebased_near_starting_capital():
    """Each window's equity[0] is starting_capital * (1 + r0), close to start."""
    data = _make_data(500)
    result = run_backtest(_buy_and_hold, data, starting_capital=100_000.0)
    oos = split_result(result, split=0.7)
    is_first = oos.in_sample.result.equity.iloc[0]
    oos_first = oos.out_of_sample.result.equity.iloc[0]
    assert 0.95 * 100_000 < is_first < 1.05 * 100_000
    assert 0.95 * 100_000 < oos_first < 1.05 * 100_000


def test_compound_consistency_is_times_oos_equals_full():
    """IS_compound * OOS_compound == Full_compound (math invariant).

    The window equities are rebuilt from the same per-bar returns as the
    original full backtest, so multiplying the two compound factors must
    equal the full compound factor to floating-point tolerance. This is the
    most important integrity check: it proves the slicing doesn't lose or
    invent any return.
    """
    data = _make_data(800)
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split=0.5)

    full_factor = result.equity.iloc[-1] / result.starting_capital
    is_factor = oos.in_sample.result.equity.iloc[-1] / result.starting_capital
    oos_factor = oos.out_of_sample.result.equity.iloc[-1] / result.starting_capital

    assert is_factor * oos_factor == pytest.approx(full_factor, rel=1e-10)


def test_returns_window_total_matches_slice_of_full():
    """Per-bar returns in each window are exactly the slice of the full series."""
    data = _make_data(500)
    result = run_backtest(_buy_and_hold, data)
    oos = split_result(result, split=0.7)

    pd.testing.assert_series_equal(
        oos.in_sample.result.returns,
        result.returns.iloc[: oos.split_index],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        oos.out_of_sample.result.returns,
        result.returns.iloc[oos.split_index :],
        check_names=False,
    )


# --- input validation -----------------------------------------------------


def test_split_fraction_out_of_range_raises():
    data = _make_data(500)
    result = run_backtest(_buy_and_hold, data)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split_result(result, split=bad)


def test_short_data_raises():
    data = _make_data(3)
    result = run_backtest(_buy_and_hold, data)
    with pytest.raises(ValueError):
        split_result(result, split=0.5)


def test_split_outside_index_range_raises():
    data = _make_data(500)
    result = run_backtest(_buy_and_hold, data)
    with pytest.raises(ValueError):
        split_result(result, split="1999-01-01")  # before data
    with pytest.raises(ValueError):
        split_result(result, split="2099-01-01")  # after data


def test_unsupported_split_type_raises():
    data = _make_data(500)
    result = run_backtest(_buy_and_hold, data)
    with pytest.raises(TypeError):
        split_result(result, split=[1, 2, 3])  # type: ignore[arg-type]


# --- trades ---------------------------------------------------------------


def test_trades_filtered_to_window_no_boundary_double_counting():
    """Trades count in exactly one window, or are excluded if they straddle."""
    data = _make_data(900)
    result = run_backtest(_flip_strategy, data)
    oos = split_result(result, split=0.5)

    is_trades = oos.in_sample.result.trades
    oos_trades = oos.out_of_sample.result.trades

    # Every IS trade fully sits inside the IS window.
    for t in is_trades:
        assert oos.in_sample.start <= t.entry_time
        assert t.exit_time <= oos.in_sample.end
    # Every OOS trade fully sits inside the OOS window.
    for t in oos_trades:
        assert oos.out_of_sample.start <= t.entry_time
        assert t.exit_time <= oos.out_of_sample.end

    # Boundary trades (entered in IS, exited in OOS) are dropped from both.
    boundary_trades = [
        t for t in result.trades
        if t.entry_time <= oos.in_sample.end and t.exit_time > oos.in_sample.end
    ]
    assert len(is_trades) + len(oos_trades) == len(result.trades) - len(boundary_trades)
