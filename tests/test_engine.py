"""Unit tests for the backtest engine, cost model, validator, and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.engine.contract import POSITION_MAX, POSITION_MIN
from backend.engine.costs import CostConfig, compute_costs
from backend.engine import metrics as M
from backend.engine.validator import validate_strategy
from backend.engine.vectorized import BacktestResult, run_backtest


# --- Fixtures --------------------------------------------------------------

def _flat_data(n: int = 252, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC", name="timestamp")
    rets = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


def _trending_data(n: int = 500) -> pd.DataFrame:
    """Deterministic upward trend - useful for sanity tests."""
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC", name="timestamp")
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


# --- THE critical test ----------------------------------------------------

def test_look_ahead_bias_is_neutralized():
    """
    THE MOST IMPORTANT TEST IN THE PROJECT.

    Construct a 'cheating' strategy that knows the same bar's close return
    perfectly. Without any guard, the cheater's Sharpe is astronomical
    (sign(r) * r = |r| has a Sharpe in the tens). With the shift, the
    foreknowledge becomes one-bar-stale; on i.i.d. random data the edge
    collapses to noise.

    We test the guard's *effect* (relative reduction) rather than an absolute
    Sharpe threshold, because at finite N the post-shift Sharpe is itself a
    noisy quantity (std error ~ sqrt(252/N) ~ 0.5 at N=1000). A >=10x
    reduction is a strong, seed-robust guarantee that the guard is doing work.
    """
    data = _flat_data(n=1000, seed=42)
    same_bar_returns = data["close"].pct_change().fillna(0.0)

    def cheating_strategy(d: pd.DataFrame, params: dict) -> pd.Series:
        # Position perfectly aligned with same-bar return: future-snooping.
        return np.sign(same_bar_returns).clip(POSITION_MIN, POSITION_MAX)

    # Zero costs so we isolate the look-ahead effect alone.
    zero_costs = CostConfig(slippage_bps=0, commission_bps=0, fx_bps=0)
    result = run_backtest(cheating_strategy, data, cost_config=zero_costs)
    sh_with_guard = M.sharpe(result.returns, bars_per_year=252)

    # What the cheater would achieve WITHOUT any shift: position[t] = sign(r[t]),
    # so strat_ret[t] = sign(r[t]) * r[t] = |r[t]|, which has a sky-high Sharpe.
    cheater_no_guard_returns = same_bar_returns * np.sign(same_bar_returns)
    sh_no_guard = M.sharpe(cheater_no_guard_returns, bars_per_year=252)

    # Sanity: without the guard, the cheater obviously prints money.
    assert sh_no_guard > 10.0, (
        f"Test setup broken: unshifted cheater should have Sharpe>>10; got {sh_no_guard:.2f}."
    )

    # The guard must cut the cheater's Sharpe by at least 10x.
    assert abs(sh_with_guard) < sh_no_guard / 10.0, (
        f"Look-ahead guard appears broken: cheater achieved Sharpe={sh_with_guard:.2f} "
        f"WITH the guard vs {sh_no_guard:.2f} without. Expected >=10x reduction. "
        f"This is the project's most critical invariant - investigate vectorized.run_backtest."
    )


def test_look_ahead_guard_uses_unshifted_in_raw_positions():
    """The result exposes both raw and shifted positions; raw must equal what the strategy returned."""
    data = _flat_data(n=100)
    raw_signal = pd.Series(0.5, index=data.index)

    def s(d, p):
        return raw_signal.copy()

    result = run_backtest(s, data, cost_config=CostConfig(slippage_bps=0, fx_bps=0))
    pd.testing.assert_series_equal(result.raw_positions, raw_signal, check_names=False)
    # Shifted must be raw shifted by 1 with leading zero.
    expected_shifted = raw_signal.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(result.positions, expected_shifted, check_names=False)


# --- Cost model ------------------------------------------------------------

def test_zero_costs_have_zero_impact():
    """With zero costs, a buy-and-hold backtest must equal close.pct_change exactly."""
    data = _trending_data(n=200)
    def buy_and_hold(d, p): return pd.Series(1.0, index=d.index)

    zero = CostConfig(slippage_bps=0, commission_bps=0, fx_bps=0, instrument_is_gbp=True)
    result = run_backtest(buy_and_hold, data, cost_config=zero)

    raw_returns = data["close"].pct_change().fillna(0.0)
    # Position is 1.0 from bar 1 onwards (shifted), so strat_ret matches from bar 1.
    pd.testing.assert_series_equal(
        result.returns.iloc[1:], raw_returns.iloc[1:], check_names=False
    )


def test_high_turnover_costs_decline_equity():
    """A whipsaw strategy with 100% daily turnover at 100 bps should bleed equity."""
    data = _flat_data(n=252, seed=7)
    # Alternate +1 / 0 every bar = 100% turnover daily.
    alt = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(len(data))], index=data.index)
    def s(d, p): return alt

    high_cost = CostConfig(slippage_bps=100, commission_bps=0, fx_bps=0, instrument_is_gbp=True)
    result = run_backtest(s, data, cost_config=high_cost)

    # 100 bps * ~0.5 turnover average * 252 bars >> any plausible drift => equity declines.
    assert result.equity.iloc[-1] < result.equity.iloc[0]


def test_borrow_cost_only_applied_when_short():
    """Borrow cost must be zero when there are no short positions."""
    data = _flat_data(n=50)
    long_only = pd.Series(1.0, index=data.index)
    cfg = CostConfig(slippage_bps=0, commission_bps=0, fx_bps=0,
                     borrow_bps_annual=1000, instrument_is_gbp=True)
    costs = compute_costs(long_only, data, cfg)
    # Only the first bar has turnover (0 -> 1); after that, no costs at all.
    assert (costs.iloc[1:] == 0).all()


# --- Metrics ---------------------------------------------------------------

def test_metrics_known_equity_curve():
    """Hand-construct a curve and verify metrics are self-consistent."""
    idx = pd.bdate_range("2020-01-01", periods=252, tz="UTC")
    # Equity that doubles over 1 year with smooth growth.
    equity = pd.Series(np.linspace(100_000, 200_000, 252), index=idx)
    rets = equity.pct_change().fillna(0.0)
    dd = equity / equity.cummax() - 1.0  # zero throughout

    assert M.total_return(equity) == pytest.approx(1.0, rel=1e-6)
    assert M.cagr(equity, 252) == pytest.approx(1.0, rel=1e-2)  # ~doubled in ~1 year
    assert M.max_drawdown(dd) == 0.0
    assert M.annualized_vol(rets, 252) > 0
    assert M.sharpe(rets, 252) > 0


def test_metrics_zero_returns():
    rets = pd.Series([0.0] * 252, index=pd.bdate_range("2020-01-01", periods=252, tz="UTC"))
    assert M.sharpe(rets, 252) == 0.0
    assert M.annualized_vol(rets, 252) == 0.0


def test_max_drawdown_known():
    """100 -> 50 -> 75: max DD is -50%."""
    idx = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
    equity = pd.Series([100, 50, 75], index=idx, dtype=float)
    dd = equity / equity.cummax() - 1.0
    assert M.max_drawdown(dd) == pytest.approx(-0.5)


def test_drawdown_duration():
    """Equity peaks at bar 0, recovers at bar 4: duration = 4."""
    idx = pd.bdate_range("2020-01-01", periods=5, tz="UTC")
    equity = pd.Series([100, 90, 80, 90, 100], index=idx, dtype=float)
    assert M.max_drawdown_duration_bars(equity) == 3  # bars 1,2,3 are underwater


# --- Edge cases ------------------------------------------------------------

def test_all_zero_positions():
    data = _flat_data(n=100)
    def flat(d, p): return pd.Series(0.0, index=d.index)
    result = run_backtest(flat, data, cost_config=CostConfig(slippage_bps=0, fx_bps=0))
    assert (result.returns == 0).all()
    assert result.equity.iloc[-1] == result.starting_capital
    assert len(result.trades) == 0


def test_positions_clipped_to_bounds():
    data = _flat_data(n=50)
    def overshoot(d, p): return pd.Series(5.0, index=d.index)  # well above POSITION_MAX
    result = run_backtest(overshoot, data, cost_config=CostConfig(slippage_bps=0, fx_bps=0,
                                                                   instrument_is_gbp=True))
    assert (result.positions <= POSITION_MAX + 1e-9).all()


def test_data_missing_columns_raises():
    bad = pd.DataFrame({"close": [1, 2, 3]},
                       index=pd.bdate_range("2020-01-01", periods=3, tz="UTC"))
    def s(d, p): return pd.Series(1.0, index=d.index)
    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest(s, bad)


# --- Validator -------------------------------------------------------------

def test_validator_accepts_clean_strategy():
    src = """
def strategy(data, params):
    return (data['close'] * 0 + 1.0)
"""
    result = validate_strategy(src)
    assert result.ok, f"Errors: {result.errors}"


def test_validator_rejects_banned_import():
    src = """
import os
def strategy(data, params):
    return data['close'] * 0
"""
    result = validate_strategy(src)
    assert not result.ok
    assert any("os" in e for e in result.errors)


def test_validator_rejects_eval():
    src = """
def strategy(data, params):
    return eval("data['close'] * 0")
"""
    result = validate_strategy(src)
    assert not result.ok
    assert any("eval" in e for e in result.errors)


def test_validator_rejects_out_of_bounds_positions():
    src = """
def strategy(data, params):
    return data['close'] * 0 + 5.0  # 5.0 is way out of range
"""
    result = validate_strategy(src)
    assert not result.ok
    assert any("out of bounds" in e for e in result.errors)


def test_validator_rejects_missing_strategy_function():
    src = """
def something_else(data, params):
    return data['close'] * 0
"""
    result = validate_strategy(src)
    assert not result.ok


def test_validator_rejects_dunder_attribute():
    src = """
def strategy(data, params):
    return (data['close'] * 0).__class__
"""
    result = validate_strategy(src)
    assert not result.ok
