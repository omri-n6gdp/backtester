"""Tests for walk-forward optimization (backend.engine.robustness.walk_forward)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.engine.robustness.walk_forward import (
    WalkForwardResult,
    _expand_grid,
    _select_best,
    walk_forward,
)


# --- fixtures -------------------------------------------------------------


def _make_data(n_bars: int = 500, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV with positive-drift GBM."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0006, 0.012, n_bars)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.005, n_bars))
    low = close * (1 - rng.uniform(0, 0.005, n_bars))
    open_ = close * (1 + rng.uniform(-0.003, 0.003, n_bars))
    volume = rng.uniform(1e5, 1e7, n_bars)
    idx = pd.date_range("2010-01-01", periods=n_bars, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _const_pos_strategy(value):
    """Factory: returns a strategy that holds a constant target position."""
    def strat(data, params):
        return pd.Series(value, index=data.index)
    return strat


def _param_dispatch_strategy(data, params):
    """Strategy whose position equals params['size'] (must be in [-1, +1])."""
    return pd.Series(params["size"], index=data.index)


# --- grid expansion -------------------------------------------------------


def test_grid_expansion_dict_form_cartesian():
    combos = _expand_grid({"a": [1, 2], "b": [10, 20]})
    assert len(combos) == 4
    assert {"a": 1, "b": 10} in combos
    assert {"a": 1, "b": 20} in combos
    assert {"a": 2, "b": 10} in combos
    assert {"a": 2, "b": 20} in combos


def test_grid_expansion_with_scalar_value_treated_as_singleton():
    combos = _expand_grid({"a": 1, "b": [10, 20]})
    assert combos == [{"a": 1, "b": 10}, {"a": 1, "b": 20}]


def test_grid_expansion_dict_form_is_deterministic():
    a = _expand_grid({"x": [1, 2, 3], "y": [10, 20]})
    b = _expand_grid({"y": [10, 20], "x": [1, 2, 3]})
    assert a == b  # key order doesn't matter; sorted internally


def test_grid_expansion_list_form_passes_through():
    grid = [{"a": 1, "b": 10}, {"a": 2, "b": 20}]
    combos = _expand_grid(grid)
    assert combos == grid
    # Returned list is a copy (not the same object).
    assert combos is not grid


def test_grid_expansion_invalid_inputs_raise():
    with pytest.raises(ValueError):
        _expand_grid({})
    with pytest.raises(ValueError):
        _expand_grid([])
    with pytest.raises(ValueError):
        _expand_grid({"a": []})  # empty value list
    with pytest.raises(ValueError):
        _expand_grid([{"a": 1}, "not a dict"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        _expand_grid("not a grid")  # type: ignore[arg-type]


# --- selection helper -----------------------------------------------------


def test_select_best_higher_is_better():
    assert _select_best([1.0, 3.0, 2.0], higher_is_better=True) == 1


def test_select_best_lower_is_better():
    assert _select_best([1.0, 3.0, 2.0], higher_is_better=False) == 0


def test_select_best_skips_nan():
    assert _select_best([float("nan"), 0.5, 0.7], higher_is_better=True) == 2


def test_select_best_all_nan_returns_none():
    assert _select_best([float("nan"), float("nan")], higher_is_better=True) is None


# --- window scheduling ----------------------------------------------------


def test_window_scheduling_correct_count_and_indices():
    """Count and per-window indices match the documented formula."""
    data = _make_data(200)
    grid = {"size": [1.0]}  # single combo, no real selection needed
    res = walk_forward(_param_dispatch_strategy, data, grid, train_bars=60, test_bars=30)

    expected_n = (200 - 60) // 30  # 4
    assert len(res.windows) == expected_n

    for k, w in enumerate(res.windows):
        train_start = k * 30
        train_end = train_start + 60       # exclusive in slicing
        test_start = train_end
        test_end = test_start + 30          # exclusive in slicing

        assert w.window_index == k
        assert w.n_train_bars == 60
        assert w.n_test_bars == 30
        assert w.train_start == data.index[train_start]
        assert w.train_end == data.index[train_end - 1]
        assert w.test_start == data.index[test_start]
        assert w.test_end == data.index[test_end - 1]


def test_train_window_does_not_see_test_bars():
    """Keystone: during selection, the strategy only ever sees training bars.

    We instrument the strategy to record the index of every slice it's
    called with, then verify (a) the training calls per window cover only
    that window's training indices and (b) the test call covers only that
    window's test indices.
    """
    n = 200
    train_bars = 60
    test_bars = 30
    n_windows = (n - train_bars) // test_bars  # 4
    n_combos = 2

    data = _make_data(n)
    seen_indices: list[set] = []

    def spy_strategy(d, params):
        seen_indices.append(set(d.index))
        return pd.Series(0.0, index=d.index)

    walk_forward(
        spy_strategy, data, {"x": [1, 2]},
        train_bars=train_bars, test_bars=test_bars,
    )

    # Per window: n_combos training calls + 1 test call.
    expected_total = n_windows * (n_combos + 1)
    assert len(seen_indices) == expected_total

    for k in range(n_windows):
        train_start = k * test_bars
        train_end = train_start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        train_set = set(data.index[train_start:train_end])
        test_set = set(data.index[test_start:test_end])

        offset = k * (n_combos + 1)
        # n_combos training calls — each gets the training slice and ONLY
        # the training slice (no test bars leaked in).
        for j in range(n_combos):
            seen = seen_indices[offset + j]
            assert seen == train_set, f"window {k} combo {j}: wrong slice"
            assert seen.isdisjoint(test_set), f"window {k} combo {j}: test bar leaked!"
        # The post-selection test call gets exactly the test slice.
        assert seen_indices[offset + n_combos] == test_set


# --- concatenation math ---------------------------------------------------


def test_concatenated_oos_equity_is_chained_returns_product():
    """oos_equity[i] == starting_capital * prod(1 + r) for r in oos_returns up to i."""
    data = _make_data(300)
    res = walk_forward(
        _param_dispatch_strategy, data, {"size": [1.0]},
        train_bars=100, test_bars=50, starting_capital=100_000.0,
    )
    expected = (1.0 + res.oos_returns).cumprod() * 100_000.0
    pd.testing.assert_series_equal(res.oos_equity, expected, check_names=False)


def test_oos_equity_index_is_concatenation_of_test_windows():
    """The OOS index is the union of all test windows' indices, sorted, deduped."""
    data = _make_data(300)
    res = walk_forward(
        _param_dispatch_strategy, data, {"size": [1.0]},
        train_bars=100, test_bars=50,
    )
    expected_index = []
    for w in res.windows:
        # Reconstruct each test window's index from train_end / test_start / test_end.
        mask = (data.index >= w.test_start) & (data.index <= w.test_end)
        expected_index.append(data.index[mask])
    expected = pd.DatetimeIndex(np.concatenate([idx.values for idx in expected_index]))
    expected = pd.DatetimeIndex(expected, tz=data.index.tz)

    assert list(res.oos_equity.index) == list(expected)
    assert res.oos_equity.index.is_monotonic_increasing
    assert not res.oos_equity.index.duplicated().any()


# --- selection-metric semantics -------------------------------------------


def test_selection_metric_drives_selection():
    """Different (selection_metric, higher_is_better) configs select differently.

    For a constant-position strategy, sharpe is approximately invariant to
    position scaling (numerator and denominator both scale linearly with
    |position|), so {size=0.5, size=1.0} is a degenerate test of sharpe
    selection. Instead, use {mode=0 (flat), mode=1 (long)}:

      mode=0 (flat) -> sharpe=0 (zero std handled in metric), vol=0.
      mode=1 (long) -> sharpe = sign(drift) * |drift|/std, vol > 0.

    Vol-min must ALWAYS pick mode=0 (zero is unambiguously the min).
    Sharpe-max must pick mode=1 on at least one bullish IS window
    (positive-drift seed makes this near-certain).
    """
    def long_or_flat(data, params):
        return pd.Series(float(params["mode"]), index=data.index)

    data = _make_data(400)
    grid = {"mode": [0, 1]}

    # Vol-min: deterministic — flat wins every time.
    res_vol = walk_forward(
        long_or_flat, data, grid,
        train_bars=120, test_bars=40,
        selection_metric="annualized_vol", higher_is_better=False,
    )
    assert all(w.selected_params["mode"] == 0 for w in res_vol.windows)

    # Sharpe-max: at least one bullish IS window picks long.
    res_sharpe = walk_forward(
        long_or_flat, data, grid,
        train_bars=120, test_bars=40,
        selection_metric="sharpe", higher_is_better=True,
    )
    assert any(w.selected_params["mode"] == 1 for w in res_sharpe.windows), (
        "Expected at least one window to pick mode=1 (long) under Sharpe selection. "
        "If this fails, the seeded data has only bearish IS windows."
    )


# --- param stability ------------------------------------------------------


def test_param_stability_counts_sum_to_n_windows():
    data = _make_data(400)
    res = walk_forward(
        _param_dispatch_strategy, data, {"size": [0.5, 1.0]},
        train_bars=120, test_bars=40,
    )
    n_windows = len(res.windows)
    assert n_windows > 0
    counts = res.param_stability["size"]["counts"]
    assert sum(counts.values()) == n_windows
    assert res.param_stability["size"]["n_distinct"] >= 1
    assert res.param_stability["size"]["mode"] in {0.5, 1.0}


# --- aggregate metrics ----------------------------------------------------


def test_aggregate_oos_metrics_are_finite_and_consistent():
    """Aggregate metrics are computed on the concatenated curve and are finite."""
    data = _make_data(400)
    res = walk_forward(
        _param_dispatch_strategy, data, {"size": [1.0]},
        train_bars=120, test_bars=40,
    )
    # cagr, sharpe, max_drawdown should all be finite real numbers.
    assert math.isfinite(res.oos_metrics.cagr)
    assert math.isfinite(res.oos_metrics.sharpe)
    assert res.oos_metrics.max_drawdown <= 0.0


def test_n_combinations_recorded_correctly():
    data = _make_data(300)
    res = walk_forward(
        _param_dispatch_strategy, data, {"size": [0.25, 0.5, 0.75, 1.0]},
        train_bars=100, test_bars=50,
    )
    assert res.n_combinations == 4


# --- input validation -----------------------------------------------------


def test_insufficient_bars_raises():
    data = _make_data(50)  # smaller than 60 + 30
    with pytest.raises(ValueError):
        walk_forward(
            _param_dispatch_strategy, data, {"size": [1.0]},
            train_bars=60, test_bars=30,
        )


def test_invalid_selection_metric_raises():
    data = _make_data(200)
    with pytest.raises(ValueError):
        walk_forward(
            _param_dispatch_strategy, data, {"size": [1.0]},
            train_bars=60, test_bars=30,
            selection_metric="not_a_real_metric",
        )


def test_negative_window_sizes_raise():
    data = _make_data(200)
    with pytest.raises(ValueError):
        walk_forward(
            _param_dispatch_strategy, data, {"size": [1.0]},
            train_bars=0, test_bars=30,
        )
    with pytest.raises(ValueError):
        walk_forward(
            _param_dispatch_strategy, data, {"size": [1.0]},
            train_bars=60, test_bars=-1,
        )
