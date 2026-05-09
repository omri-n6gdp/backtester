"""
Unit tests for backend/engine/robustness/monte_carlo.py.

Uses duck-typed BacktestResult mocks (FakeResult / FakeTrade) so these tests
have zero dependency on the rest of the engine. The MC module only needs
.equity (sliceable), and .trades (iterable of objects with .pnl_return).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from backend.engine.robustness.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    monte_carlo_shuffle,
    _drawdown_duration_in_trades,
    _quantile_key,
)


# ----- fixtures -------------------------------------------------------------

@dataclass
class FakeTrade:
    pnl_return: float


@dataclass
class FakeResult:
    equity: pd.Series
    trades: list[FakeTrade]


def make_result(trade_returns: Iterable[float], starting_capital: float = 100_000.0) -> FakeResult:
    rs = list(trade_returns)
    n = max(len(rs), 1)
    idx = pd.date_range("2020-01-01", periods=n + 1, freq="D", tz="UTC")
    if rs:
        eq_inner = starting_capital * np.cumprod(1.0 + np.array(rs))
        eq = np.concatenate(([starting_capital], eq_inner))
    else:
        eq = np.full(n + 1, starting_capital)
    return FakeResult(equity=pd.Series(eq, index=idx), trades=[FakeTrade(pnl_return=r) for r in rs])


# ----- helpers --------------------------------------------------------------

def test_quantile_key_format() -> None:
    assert _quantile_key(0.05) == "q05"
    assert _quantile_key(0.5) == "q50"
    assert _quantile_key(0.95) == "q95"
    assert _quantile_key(0.0) == "q00"
    assert _quantile_key(1.0) == "q100"


def test_drawdown_duration_handles_edge_cases() -> None:
    assert _drawdown_duration_in_trades(np.array([])) == 0
    assert _drawdown_duration_in_trades(np.array([0.0, 0.0, 0.0])) == 0
    assert _drawdown_duration_in_trades(np.array([0.0, -0.1, -0.2, 0.0])) == 2
    assert _drawdown_duration_in_trades(np.array([-0.1, 0.0, -0.2, -0.3, 0.0, -0.1])) == 2


# ----- config validation ----------------------------------------------------

def test_config_rejects_zero_iterations() -> None:
    with pytest.raises(ValueError, match="n_iterations"):
        MonteCarloConfig(n_iterations=0)


def test_config_rejects_bad_quantiles() -> None:
    with pytest.raises(ValueError, match="quantile"):
        MonteCarloConfig(quantiles=(0.5, 1.2))
    with pytest.raises(ValueError, match="quantile"):
        MonteCarloConfig(quantiles=(-0.1, 0.5))
    with pytest.raises(ValueError, match="quantile"):
        MonteCarloConfig(quantiles=())


# ----- engine semantics -----------------------------------------------------

def test_no_trades_returns_degenerate_result() -> None:
    res = make_result([])
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=10))
    assert out.n_trades == 0
    assert out.n_iterations == 10
    assert all(eq == 100_000.0 for eq in out.final_equities)
    assert all(dd == 0.0 for dd in out.max_drawdowns)
    assert math.isnan(out.original_dd_percentile)
    for band in out.equity_bands.values():
        assert band == [100_000.0]
    for band in out.drawdown_bands.values():
        assert band == [0.0]


def test_single_trade_does_not_crash() -> None:
    res = make_result([0.05])
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=50))
    assert out.n_trades == 1
    # Only one permutation possible -> all shuffles identical
    assert len(set(out.final_equities)) == 1
    assert len(set(out.max_drawdowns)) == 1
    # Single positive trade -> no drawdown
    assert max(out.max_drawdowns) == 0.0


def test_determinism_same_seed_same_output() -> None:
    res = make_result([0.05, -0.03, 0.02, -0.01, 0.04, -0.06, 0.03])
    a = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=200, seed=123))
    b = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=200, seed=123))
    assert a.final_equities == b.final_equities
    assert a.max_drawdowns == b.max_drawdowns
    assert a.equity_bands == b.equity_bands


def test_different_seeds_differ() -> None:
    res = make_result([0.05, -0.03, 0.02, -0.01, 0.04, -0.06, 0.03])
    a = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=200, seed=1))
    b = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=200, seed=2))
    # Final equities are mathematically equal across all shuffles (permutation
    # invariance of prod), so they match regardless of seed; max DDs are
    # order-dependent and must differ.
    assert a.max_drawdowns != b.max_drawdowns


def test_final_equity_is_invariant_under_shuffling() -> None:
    """Permutation invariance of prod(1+r). Single most important sanity check."""
    rng = np.random.default_rng(0)
    rs = rng.normal(0.001, 0.02, 50).tolist()
    res = make_result(rs, starting_capital=100_000.0)
    expected = 100_000.0 * float(np.prod(1.0 + np.array(rs)))
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=500, seed=7))
    arr = np.array(out.final_equities)
    assert np.allclose(arr, expected, rtol=1e-10, atol=1e-6)
    assert math.isclose(out.original_final_equity, expected, rel_tol=1e-12)


def test_quantile_bands_are_ordered() -> None:
    """At every step, q05 <= q50 <= q95 (both equity and drawdown sides)."""
    rng = np.random.default_rng(0)
    rs = rng.normal(0.001, 0.02, 30).tolist()
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=500, seed=11))
    q05 = np.array(out.equity_bands["q05"])
    q50 = np.array(out.equity_bands["q50"])
    q95 = np.array(out.equity_bands["q95"])
    assert np.all(q05 <= q50 + 1e-9)
    assert np.all(q50 <= q95 + 1e-9)
    d05 = np.array(out.drawdown_bands["q05"])
    d50 = np.array(out.drawdown_bands["q50"])
    d95 = np.array(out.drawdown_bands["q95"])
    assert np.all(d05 <= d50 + 1e-9)
    assert np.all(d50 <= d95 + 1e-9)


def test_drawdown_bands_are_nonpositive() -> None:
    rng = np.random.default_rng(0)
    rs = rng.normal(0.001, 0.02, 30).tolist()
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=200, seed=11))
    for key, band in out.drawdown_bands.items():
        arr = np.array(band)
        assert np.all(arr <= 1e-12), f"band {key} has positive values"


def test_band_lengths_match_n_trades_plus_one() -> None:
    rs = [0.01] * 17
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=100, seed=0))
    for band in out.equity_bands.values():
        assert len(band) == 17 + 1  # includes the starting bar
    for band in out.drawdown_bands.values():
        assert len(band) == 17 + 1


def test_dd_percentile_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    rs = rng.normal(0.001, 0.02, 25).tolist()
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=300, seed=0))
    assert 0.0 <= out.original_dd_percentile <= 100.0


def test_lucky_ordering_yields_high_percentile() -> None:
    """Interleaved gains and losses produce a shallow DD because each loss
    is immediately offset by the next gain. Most random shuffles will pile
    up consecutive losses (against either the initial peak, OR a built-up
    peak from prior wins) and produce deeper DDs, so percentile is high.

    Note: 'wins first then losses' is NOT a lucky ordering — it builds a
    high peak and then drops the entire loss run against it. The truly
    shallow-DD ordering is interleaved."""
    rs = [0.10, -0.05, 0.08, -0.07, 0.06, -0.09, 0.04, -0.11, 0.02]
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=2000, seed=42))
    assert out.original_dd_percentile > 50.0


def test_unlucky_ordering_yields_low_percentile() -> None:
    """If losses come first (drawdown is deep on a low base), most shuffles
    produce a shallower DD, so percentile is low."""
    rs = [-0.11, -0.09, -0.07, -0.05, 0.02, 0.04, 0.06, 0.08, 0.10]
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=2000, seed=42))
    assert out.original_dd_percentile < 50.0


def test_max_dd_quantile_ordering() -> None:
    """q05 of max-DD distribution is more negative than q95 (q05 = worst 5%)."""
    rng = np.random.default_rng(0)
    rs = rng.normal(0.001, 0.02, 40).tolist()
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=500, seed=0))
    assert out.max_dd_quantiles["q05"] <= out.max_dd_quantiles["q50"] + 1e-12
    assert out.max_dd_quantiles["q50"] <= out.max_dd_quantiles["q95"] + 1e-12


def test_per_iteration_arrays_have_correct_length() -> None:
    rs = [0.01, -0.02, 0.03, -0.01]
    res = make_result(rs)
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=37, seed=0))
    assert len(out.final_equities) == 37
    assert len(out.max_drawdowns) == 37
    assert len(out.max_drawdown_durations) == 37


def test_custom_quantiles_are_honored() -> None:
    rs = [0.05, -0.03, 0.02, -0.01]
    res = make_result(rs)
    cfg = MonteCarloConfig(n_iterations=100, seed=0, quantiles=(0.1, 0.9))
    out = monte_carlo_shuffle(res, cfg)
    assert set(out.equity_bands.keys()) == {"q10", "q90"}
    assert set(out.drawdown_bands.keys()) == {"q10", "q90"}
    assert set(out.max_dd_quantiles.keys()) == {"q10", "q90"}
    assert out.quantiles == [0.1, 0.9]


def test_original_reference_matches_unshuffled_construction() -> None:
    rs = [0.10, -0.05, 0.03, -0.02, 0.04, -0.08]
    res = make_result(rs)
    # Compute the expected original DD by hand.
    eq = np.concatenate(([100_000.0], 100_000.0 * np.cumprod(1.0 + np.array(rs))))
    rmax = np.maximum.accumulate(eq)
    dd = (eq - rmax) / rmax
    expected_max_dd = float(dd.min())
    out = monte_carlo_shuffle(res, MonteCarloConfig(n_iterations=10, seed=0))
    assert math.isclose(out.original_max_drawdown, expected_max_dd, rel_tol=1e-12)
    assert math.isclose(out.original_final_equity, float(eq[-1]), rel_tol=1e-12)
