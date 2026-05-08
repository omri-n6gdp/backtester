"""
In-sample / out-of-sample split for honest backtest assessment.

Given a single BacktestResult run on the FULL history, slice it into IS and
OOS portions, rebase each portion's equity curve to start at
`starting_capital`, and compute a full MetricsBundle on each portion
independently.

The standard pattern: optimize parameters on the IS window, then report OOS
metrics as the honest assessment. OOS metrics are the ones that count.
This module does the slicing; parameter optimization itself is in
`walk_forward.py` (Phase 5.2).

Why slice a single full-history backtest instead of running two backtests
on subsets of the data?
  1. Strategy warmup is automatic: by the time we enter the OOS window,
     indicators (e.g. SMA(200)) are already at steady state.
  2. Per-bar returns are identical to the original full backtest, so the
     accounting is consistent: IS_compound x OOS_compound == full_compound.
  3. Walk-forward (5.2) follows the same rolling-slice pattern.

Boundary trades — trades whose entry is in IS and exit is in OOS — are
excluded from both windows' trade-level statistics, because attributing
P&L to one side or the other is ambiguous. Their per-bar P&L is still
present in the sliced returns (and therefore in equity, drawdown, Sharpe
etc.); only the trade list and trade-derived metrics drop them.

Usage:
    result = run_backtest(strategy, data, params)
    oos = split_result(result, split=0.7)              # 70/30 fraction
    oos = split_result(result, split="2020-01-01")     # split at a date
    oos.in_sample.metrics.sharpe        # the optimization target
    oos.out_of_sample.metrics.sharpe    # the honest number
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import pandas as pd

from ..metrics import MetricsBundle, compute_metrics
from ..vectorized import BacktestResult, Trade


SplitSpec = Union[float, str, pd.Timestamp]


@dataclass
class WindowResult:
    """A backtest sub-result over a contiguous time window, with metrics."""
    label: str                # 'in_sample' or 'out_of_sample'
    start: pd.Timestamp       # first bar timestamp
    end: pd.Timestamp         # last bar timestamp
    n_bars: int
    result: BacktestResult    # rebased to start at starting_capital
    metrics: MetricsBundle


@dataclass
class OOSResult:
    """Output of an in-sample / out-of-sample split."""
    split_timestamp: pd.Timestamp   # first OOS bar's timestamp
    split_index: int                # OOS starts at this index in the original
    in_sample: WindowResult
    out_of_sample: WindowResult


def split_result(
    result: BacktestResult,
    split: SplitSpec = 0.7,
) -> OOSResult:
    """
    Split a BacktestResult into IS and OOS WindowResults.

    Args:
        result: BacktestResult run on the full history.
        split:  one of:
                - float in (0, 1): fraction of bars in IS (default 0.7)
                - str (ISO date) or pd.Timestamp: split at this timestamp.
                  The first bar AT or AFTER this timestamp becomes the first
                  OOS bar.

    Returns:
        OOSResult with both windows populated.

    Raises:
        ValueError if `result` has fewer than 4 bars, the split fraction is
        out of (0,1), or the split point produces an empty IS or OOS window.
        TypeError if `split` is not float, str, or pd.Timestamp.
    """
    n = len(result.equity)
    if n < 4:
        raise ValueError(f"need at least 4 bars to split; got {n}")

    split_idx = _resolve_split_index(result.equity.index, split)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(
            f"split point produces an empty window: split_idx={split_idx}, n={n}"
        )

    split_ts = result.equity.index[split_idx]

    is_window = _slice_window(result, 0, split_idx, label="in_sample")
    oos_window = _slice_window(result, split_idx, n, label="out_of_sample")

    return OOSResult(
        split_timestamp=split_ts,
        split_index=split_idx,
        in_sample=is_window,
        out_of_sample=oos_window,
    )


# --- internals -------------------------------------------------------------


def _resolve_split_index(index: pd.DatetimeIndex, split: SplitSpec) -> int:
    """Convert a user-facing split spec into an integer index into `index`."""
    if isinstance(split, float):
        if not 0.0 < split < 1.0:
            raise ValueError(f"split fraction must be in (0, 1); got {split}")
        return int(len(index) * split)

    if isinstance(split, str):
        split = pd.Timestamp(split)

    if isinstance(split, pd.Timestamp):
        # Align tz between split and index so searchsorted compares cleanly.
        if index.tz is not None and split.tz is None:
            split = split.tz_localize(index.tz)
        elif index.tz is None and split.tz is not None:
            split = split.tz_localize(None)
        return int(index.searchsorted(split))

    raise TypeError(
        f"split must be float, str, or pd.Timestamp; got {type(split).__name__}"
    )


def _slice_window(
    result: BacktestResult,
    start_idx: int,
    end_idx: int,           # exclusive
    *,
    label: str,
) -> WindowResult:
    """
    Slice a BacktestResult to [start_idx:end_idx) and rebuild equity/drawdown.

    The per-bar `returns` series is sliced as-is (it is the source of truth
    for what happened during the window). Equity is rebuilt from the sliced
    returns starting from `result.starting_capital`, so each window's equity
    curve is self-contained. Drawdown is recomputed from the rebuilt equity.

    Trades are kept only when they BOTH entered and exited within the
    window. See module docstring for boundary-trade rationale.
    """
    returns = result.returns.iloc[start_idx:end_idx].copy()
    positions = result.positions.iloc[start_idx:end_idx].copy()
    raw_positions = result.raw_positions.iloc[start_idx:end_idx].copy()
    costs = result.costs.iloc[start_idx:end_idx].copy()

    rebased_equity = (1.0 + returns).cumprod() * result.starting_capital
    rolling_peak = rebased_equity.cummax()
    drawdown = rebased_equity / rolling_peak - 1.0

    window_start_ts = rebased_equity.index[0]
    window_end_ts = rebased_equity.index[-1]

    trades_in_window: list[Trade] = [
        t for t in result.trades
        if window_start_ts <= t.entry_time and t.exit_time <= window_end_ts
    ]

    sub_result = BacktestResult(
        equity=rebased_equity,
        drawdown=drawdown,
        returns=returns,
        positions=positions,
        raw_positions=raw_positions,
        costs=costs,
        trades=trades_in_window,
        starting_capital=result.starting_capital,
        bars_per_year=result.bars_per_year,
    )
    return WindowResult(
        label=label,
        start=window_start_ts,
        end=window_end_ts,
        n_bars=len(rebased_equity),
        result=sub_result,
        metrics=compute_metrics(sub_result),
    )
