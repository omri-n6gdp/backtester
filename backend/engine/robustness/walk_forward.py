"""
Walk-forward optimization (Phase 5.2).

A walk-forward run rolls a (training, test) window pair through the data,
optimizing parameters on each training window and reporting honest
out-of-sample performance on the matched test window. Concatenating all
test windows produces an aggregate OOS equity curve — a more honest
assessment than a single IS/OOS split because parameters are reselected
across regimes rather than fixed once.

The math:
    For window k (0-indexed):
        train_slice = data[k*test_bars : k*test_bars + train_bars]
        test_slice  = data[k*test_bars + train_bars : k*test_bars + train_bars + test_bars]
    n_windows = (len(data) - train_bars) // test_bars

    For each combo in the parameter grid, run a backtest on train_slice and
    compute the MetricsBundle. Take the field named by `selection_metric`
    and pick the argmax (or argmin if higher_is_better=False). Run that
    combo on test_slice. Record both.

    After every window, concatenate the per-bar test returns into one OOS
    series. Rebuild equity from `starting_capital`. Compute the aggregate
    MetricsBundle on that concatenated curve.

Look-ahead protection:
    The test slice is NEVER passed into the strategy during selection. The
    training backtest only ever receives `train_slice`. The keystone test
    `test_train_window_does_not_see_test_bars` proves this empirically
    by recording the indices the strategy is called with.

First-bar-zero artifact:
    The engine treats every backtest's first bar as flat (positions
    fillna(0.0), returns fillna(0.0)) — so each test window contributes a
    zero return on its first bar. Across N windows this loses N bars of
    return contribution from the OOS aggregate (~1% for 50 windows on
    5000 bars). This is consistent with "deploy fresh each window"
    semantics and is not a bug; just a thing to know when comparing
    walk-forward aggregate metrics to a single big backtest of the same
    period.

Grid forms:
    walk_forward(..., param_grid={"a": [1, 2], "b": [10, 20]}) -> 4 combos
    walk_forward(..., param_grid={"a": 1, "b": [10, 20]})       -> 2 combos (scalar OK)
    walk_forward(..., param_grid=[{"a": 1, "b": 10}])           -> 1 combo (explicit)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ..contract import StrategyFn
from ..costs import DEFAULT_COST_CONFIG, CostConfig
from ..metrics import MetricsBundle, compute_metrics
from ..vectorized import BacktestResult, run_backtest


ParamGrid = Union[dict, list]


@dataclass
class WindowReport:
    """Per-window result: which params won on IS, what they did on OOS."""
    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_bars: int
    n_test_bars: int
    selected_params: dict
    is_metrics: MetricsBundle
    oos_metrics: MetricsBundle


@dataclass
class WalkForwardResult:
    """Output of a walk-forward run.

    `param_stability` shape (per parameter key):
        {"counts":     {value: count_of_windows_selecting_it},
         "n_distinct": int,
         "mode":       value_most_often_selected}

    Note: `counts` keys are the original parameter values (int/float/str/...).
    Callers serializing to JSON must stringify them.
    """
    windows: list[WindowReport]
    oos_equity: pd.Series       # concatenated test windows, rebased to starting_capital
    oos_returns: pd.Series      # concatenated per-bar (post-cost) returns
    oos_drawdown: pd.Series     # rebuilt from oos_equity
    oos_positions: pd.Series    # SHIFTED positions actually applied (concatenated)
    oos_raw_positions: pd.Series  # UNSHIFTED positions returned by selected combos
    oos_costs: pd.Series        # per-bar costs in return units (concatenated)
    oos_metrics: MetricsBundle  # MetricsBundle of the concatenated OOS curve
    param_stability: dict[str, dict[str, Any]]
    n_combinations: int
    selection_metric: str
    higher_is_better: bool


# --- public entry point ---------------------------------------------------


def walk_forward(
    strategy: StrategyFn,
    data: pd.DataFrame,
    param_grid: ParamGrid,
    train_bars: int,
    test_bars: int,
    *,
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
    starting_capital: float = 100_000.0,
    bars_per_year: int = 252,
    selection_metric: str = "sharpe",
    higher_is_better: bool = True,
) -> WalkForwardResult:
    """Run a rolling walk-forward optimization.

    Args:
        strategy: validated StrategyFn (data, params) -> Series of positions.
        data:     OHLCV DataFrame for the FULL history (no pre-slicing needed).
        param_grid: either a dict (cartesian-expanded) or a list of dicts.
                    See module docstring for examples.
        train_bars: length of the in-sample (training) window in bars.
        test_bars:  length of the out-of-sample (test) window in bars; also
                    the step size by which the window pair rolls forward.
        cost_config, starting_capital, bars_per_year:
            forwarded to every internal `run_backtest` call.
        selection_metric: name of any field on MetricsBundle (e.g. "sharpe",
                          "cagr", "calmar", "max_drawdown", ...).
        higher_is_better: True selects argmax of selection_metric on the IS
                          window; False selects argmin (e.g. for "max_drawdown"
                          where shallower [closer to 0] is better, you'd want
                          higher_is_better=True since max_drawdown is a
                          negative number; for "annualized_vol" you'd want
                          higher_is_better=False).

    Raises:
        ValueError on bad inputs: train_bars/test_bars <= 0, insufficient
        data, empty grid, unknown selection_metric, all-NaN scoring on a
        window.
    """
    # --- validate ---------------------------------------------------------
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError(
            f"train_bars and test_bars must be > 0; got train={train_bars}, test={test_bars}"
        )
    n = len(data)
    if n < train_bars + test_bars:
        raise ValueError(
            f"need at least train_bars + test_bars = {train_bars + test_bars} bars; got {n}"
        )
    if not _is_metrics_field(selection_metric):
        valid = sorted(f.name for f in dataclasses.fields(MetricsBundle))
        raise ValueError(
            f"selection_metric {selection_metric!r} is not a field on MetricsBundle. "
            f"Valid: {valid}"
        )

    combos = _expand_grid(param_grid)
    if not combos:
        raise ValueError("param_grid expanded to zero combinations")

    n_windows = (n - train_bars) // test_bars
    if n_windows < 1:
        raise ValueError(
            f"insufficient bars: {n} bars yields 0 windows of train={train_bars}, test={test_bars}"
        )

    # --- run rolling windows ----------------------------------------------
    windows: list[WindowReport] = []
    test_results: list[BacktestResult] = []

    for k in range(n_windows):
        train_start_idx = k * test_bars
        train_end_idx = train_start_idx + train_bars
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_bars

        train_slice = data.iloc[train_start_idx:train_end_idx]
        test_slice = data.iloc[test_start_idx:test_end_idx]

        # Score every combo on the training slice.
        scores: list[float] = []
        is_metrics_per_combo: list[Optional[MetricsBundle]] = []
        for combo in combos:
            try:
                tr_result = run_backtest(
                    strategy, train_slice, combo,
                    cost_config=cost_config,
                    starting_capital=starting_capital,
                    bars_per_year=bars_per_year,
                )
                tr_metrics = compute_metrics(tr_result)
                score = float(getattr(tr_metrics, selection_metric))
            except Exception:
                tr_metrics = None
                score = float("nan")
            scores.append(score)
            is_metrics_per_combo.append(tr_metrics)

        best_idx = _select_best(scores, higher_is_better)
        if best_idx is None:
            raise ValueError(
                f"no valid combo on window {k} "
                f"(train {train_slice.index[0]} .. {train_slice.index[-1]}): "
                f"all {len(combos)} combos produced NaN/invalid {selection_metric!r}"
            )

        selected = combos[best_idx]
        is_metrics = is_metrics_per_combo[best_idx]
        assert is_metrics is not None  # by construction: we picked from the valid scores

        # Run the SELECTED combo on the test slice.
        test_result = run_backtest(
            strategy, test_slice, selected,
            cost_config=cost_config,
            starting_capital=starting_capital,
            bars_per_year=bars_per_year,
        )
        test_metrics = compute_metrics(test_result)
        test_results.append(test_result)

        windows.append(WindowReport(
            window_index=k,
            train_start=train_slice.index[0],
            train_end=train_slice.index[-1],
            test_start=test_slice.index[0],
            test_end=test_slice.index[-1],
            n_train_bars=len(train_slice),
            n_test_bars=len(test_slice),
            selected_params=dict(selected),
            is_metrics=is_metrics,
            oos_metrics=test_metrics,
        ))

    # --- aggregate --------------------------------------------------------
    aggregate = _build_aggregate_result(test_results, starting_capital, bars_per_year)
    aggregate_metrics = compute_metrics(aggregate)
    stability = _compute_param_stability(windows)

    return WalkForwardResult(
        windows=windows,
        oos_equity=aggregate.equity,
        oos_returns=aggregate.returns,
        oos_drawdown=aggregate.drawdown,
        oos_positions=aggregate.positions,
        oos_raw_positions=aggregate.raw_positions,
        oos_costs=aggregate.costs,
        oos_metrics=aggregate_metrics,
        param_stability=stability,
        n_combinations=len(combos),
        selection_metric=selection_metric,
        higher_is_better=higher_is_better,
    )


# --- internals ------------------------------------------------------------


def _is_metrics_field(name: str) -> bool:
    return name in {f.name for f in dataclasses.fields(MetricsBundle)}


def _expand_grid(grid: ParamGrid) -> list[dict]:
    """Expand a grid dict (cartesian) or list-of-dicts (passthrough) into combos.

    Dict form: each value must be either a list/tuple (its options) or a
    scalar (treated as a singleton list). Keys are sorted alphabetically
    for deterministic combo ordering.

    List form: each entry must be a dict; passes through verbatim.
    """
    if isinstance(grid, list):
        if not grid:
            raise ValueError("param_grid is an empty list")
        for i, entry in enumerate(grid):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"param_grid[{i}] must be a dict; got {type(entry).__name__}"
                )
        return [dict(d) for d in grid]

    if isinstance(grid, dict):
        if not grid:
            raise ValueError("param_grid is an empty dict")
        keys = sorted(grid.keys())
        per_key_values: list[list] = []
        for k in keys:
            v = grid[k]
            if isinstance(v, (list, tuple)):
                vlist = list(v)
                if not vlist:
                    raise ValueError(f"param_grid[{k!r}] has empty value list")
                per_key_values.append(vlist)
            else:
                per_key_values.append([v])
        combos: list[dict] = []
        for tup in product(*per_key_values):
            combos.append(dict(zip(keys, tup)))
        return combos

    raise TypeError(
        f"param_grid must be dict or list[dict]; got {type(grid).__name__}"
    )


def _select_best(scores: list[float], higher_is_better: bool) -> Optional[int]:
    """Return the index of the best valid score, or None if all are NaN/None."""
    valid_indices = [
        i for i, s in enumerate(scores)
        if s is not None and not (isinstance(s, float) and np.isnan(s))
    ]
    if not valid_indices:
        return None
    cmp = max if higher_is_better else min
    return cmp(valid_indices, key=lambda i: scores[i])


def _build_aggregate_result(
    test_results: list[BacktestResult],
    starting_capital: float,
    bars_per_year: int,
) -> BacktestResult:
    """Concatenate the per-bar series of every test window into one BacktestResult."""
    if not test_results:
        raise ValueError("no test windows to aggregate")

    all_returns = pd.concat([r.returns for r in test_results])
    all_positions = pd.concat([r.positions for r in test_results])
    all_raw_positions = pd.concat([r.raw_positions for r in test_results])
    all_costs = pd.concat([r.costs for r in test_results])
    all_trades: list = []
    for r in test_results:
        all_trades.extend(r.trades)

    equity = (1.0 + all_returns).cumprod() * starting_capital
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0

    return BacktestResult(
        equity=equity,
        drawdown=drawdown,
        returns=all_returns,
        positions=all_positions,
        raw_positions=all_raw_positions,
        costs=all_costs,
        trades=all_trades,
        starting_capital=starting_capital,
        bars_per_year=bars_per_year,
    )


def _compute_param_stability(windows: list[WindowReport]) -> dict[str, dict[str, Any]]:
    """For each param key, count selected values across windows + report mode/n_distinct."""
    if not windows:
        return {}
    keys: set = set()
    for w in windows:
        keys.update(w.selected_params.keys())

    stability: dict[str, dict[str, Any]] = {}
    for k in sorted(keys):
        values = [w.selected_params.get(k) for w in windows]
        counts: dict = {}
        for v in values:
            try:
                counts[v] = counts.get(v, 0) + 1
            except TypeError:
                # Unhashable values (rare; a list/dict in the grid). Stringify.
                counts[repr(v)] = counts.get(repr(v), 0) + 1
        mode_val = max(counts, key=lambda x: counts[x])
        stability[k] = {
            "counts": counts,
            "n_distinct": len(counts),
            "mode": mode_val,
        }
    return stability
