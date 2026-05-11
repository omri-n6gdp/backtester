"""
Regime conditional analysis.

Slices a BacktestResult across externally-defined market regimes and
reports per-regime metrics. A strategy that produces a respectable
headline Sharpe but only works in one regime is a different (and worse)
animal than one that works across all four. The breakdown surfaces that.

Two regime axes are provided out of the box:

  spy_trend  : 'bull'   when SPY > SPY's 200-day SMA
               'bear'   otherwise
  vix_level  : 'low_vol'    when VIX < 15
               'normal_vol' when 15 <= VIX <= 25
               'high_vol'   when VIX > 25

These follow the project spec. Both are computed from daily SPY/VIX series
and broadcast to the backtest's bar frequency via forward-fill, so the
analysis works regardless of whether the strategy is daily, hourly, or 5m.

Per-regime metrics interpretation
---------------------------------
For each regime label, we extract the bars where that label was active,
reconstruct an equity curve from those bars' returns alone (cumprod from
starting_capital), and compute metrics on the result. The annualized-rate
metrics (CAGR, vol, Sharpe) are interpretable as "the rate at which the
strategy would compound if it could only trade during this regime."

Trade attribution
-----------------
A trade can span a regime boundary (enter in bull, exit in bear). For the
per-regime trade count and win rate, we attribute trades by ENTRY-TIME
regime. This captures "how the strategy's entry decisions performed
under each regime" — the cleanest single rule. Trades whose return came
partly from a different regime than they opened in are still counted,
which is an imperfection that goes away in Phase 9's event-driven mode
where per-bar trade tracking is exact.

This module is pure: callers are responsible for fetching SPY and VIX
data themselves (the route layer does this with `get_prices`). Engine
tests use synthetic SPY/VIX series, no network required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backend.engine.metrics import MetricsBundle, compute_metrics
from backend.engine.vectorized import BacktestResult

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RegimeConfig:
    spy_trend_window: int = 200
    vix_low_threshold: float = 15.0
    vix_high_threshold: float = 25.0


@dataclass
class RegimeWindow:
    """Per-regime slice of a backtest. `metrics` is computed over the bars
    where this regime label was active, with equity rebased to
    starting_capital so the numbers compare cleanly across regimes."""
    label: str
    n_bars: int
    fraction: float           # n_bars / total_active_bars (excluding unknown)
    metrics: MetricsBundle


@dataclass
class RegimeAxisResult:
    """One axis of regime analysis (e.g. SPY trend, VIX level)."""
    name: str                       # 'spy_trend' or 'vix_level'
    n_bars_total: int
    n_bars_unknown: int             # warmup bars + bars with missing regime data
    windows: dict[str, RegimeWindow]


@dataclass
class RegimeAnalysisResult:
    """All available regime axes for one backtest. Either may be None
    if the underlying data fetch failed; errors are surfaced separately."""
    spy_trend: RegimeAxisResult | None = None
    vix_level: RegimeAxisResult | None = None
    spy_trend_error: str | None = None
    vix_level_error: str | None = None


# --- Regime label generators ----------------------------------------------

def compute_spy_trend_regime(spy_close: pd.Series, window: int = 200) -> pd.Series:
    """Bull when close > rolling SMA(window); bear when close <= SMA.
    Bars before the SMA window completes are NaN (insufficient history)."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    sma = spy_close.rolling(window).mean()
    out = pd.Series(np.where(spy_close > sma, "bull", "bear"),
                    index=spy_close.index, dtype=object)
    out[sma.isna()] = np.nan  # warmup bars
    return out


def compute_vix_regime(
    vix_close: pd.Series,
    low_threshold: float = 15.0,
    high_threshold: float = 25.0,
) -> pd.Series:
    """Three-level VIX regime. Boundaries are inclusive in the middle band."""
    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be < high_threshold ({high_threshold})"
        )
    out = pd.Series(index=vix_close.index, dtype=object)
    out[vix_close < low_threshold] = "low_vol"
    out[(vix_close >= low_threshold) & (vix_close <= high_threshold)] = "normal_vol"
    out[vix_close > high_threshold] = "high_vol"
    out[vix_close.isna()] = np.nan
    return out


# --- Partitioner ----------------------------------------------------------

def partition_by_regime(
    result: BacktestResult,
    regime_series: pd.Series,
    axis_name: str,
) -> RegimeAxisResult:
    """Split a BacktestResult by regime and compute metrics per slice.

    `regime_series` is broadcast to `result.returns.index` via forward-fill,
    so a daily regime series can label intraday bars correctly. Bars where
    the regime label is NaN (warmup, missing data) are counted as 'unknown'
    and excluded from per-regime windows.
    """
    target_index = result.returns.index
    n_total = len(target_index)

    # Align to the result's index. ffill propagates daily labels across
    # any higher-frequency bars within the same day. Bars before the first
    # known label remain NaN.
    aligned = regime_series.reindex(target_index, method="ffill")

    unknown_mask = aligned.isna()
    n_unknown = int(unknown_mask.sum())

    windows: dict[str, RegimeWindow] = {}
    n_active = n_total - n_unknown

    for label in pd.unique(aligned.dropna()):
        label_str = str(label)
        mask = (aligned == label).to_numpy()
        n_bars = int(mask.sum())
        if n_bars < 2:
            # Need at least 2 bars to compute meaningful metrics.
            continue

        # Per-regime equity: cumprod of just this regime's bars, rebased.
        regime_returns = result.returns[mask].copy()
        regime_costs = result.costs[mask].copy() if len(result.costs) else pd.Series(dtype=float)
        regime_positions = result.positions[mask].copy()
        regime_raw_pos = result.raw_positions[mask].copy() if len(result.raw_positions) else pd.Series(dtype=float)
        regime_equity = (1.0 + regime_returns).cumprod() * result.starting_capital
        regime_running_max = regime_equity.cummax()
        regime_drawdown = regime_equity / regime_running_max - 1.0

        # Trades attributed by ENTRY-TIME regime. See module docstring for
        # the rationale and limitations.
        regime_trades = []
        for t in result.trades:
            entry_label = _label_at_timestamp(aligned, t.entry_time)
            if entry_label == label:
                regime_trades.append(t)

        sub_result = BacktestResult(
            equity=regime_equity,
            drawdown=regime_drawdown,
            returns=regime_returns,
            positions=regime_positions,
            raw_positions=regime_raw_pos,
            costs=regime_costs,
            trades=regime_trades,
            starting_capital=result.starting_capital,
            bars_per_year=result.bars_per_year,
        )
        metrics = compute_metrics(sub_result)
        fraction = float(n_bars / n_active) if n_active > 0 else 0.0
        windows[label_str] = RegimeWindow(
            label=label_str,
            n_bars=n_bars,
            fraction=fraction,
            metrics=metrics,
        )

    return RegimeAxisResult(
        name=axis_name,
        n_bars_total=n_total,
        n_bars_unknown=n_unknown,
        windows=windows,
    )


def _label_at_timestamp(aligned: pd.Series, ts: pd.Timestamp):
    """Return the regime label active at `ts`, or NaN if before/missing.
    Uses the index since `aligned` is already ffill'd onto the result's index."""
    if ts in aligned.index:
        return aligned.loc[ts]
    pos = aligned.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return np.nan
    return aligned.iloc[pos]


# --- Top-level orchestrator -----------------------------------------------

def analyze_regimes(
    result: BacktestResult,
    spy_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    config: RegimeConfig | None = None,
) -> RegimeAnalysisResult:
    """Run both regime axes against `result`. Pass None for either price
    series to skip that axis (useful when fetch fails at the route layer).

    `spy_close` / `vix_close` should be daily close prices. Indexes don't
    need to match `result.returns.index` — the partitioner reindexes."""
    cfg = config or RegimeConfig()
    out = RegimeAnalysisResult()

    if spy_close is not None and len(spy_close):
        try:
            spy_regime = compute_spy_trend_regime(spy_close, window=cfg.spy_trend_window)
            out.spy_trend = partition_by_regime(result, spy_regime, axis_name="spy_trend")
        except Exception as e:  # noqa: BLE001
            out.spy_trend_error = f"{type(e).__name__}: {e}"
    elif spy_close is None:
        out.spy_trend_error = "SPY data not provided"

    if vix_close is not None and len(vix_close):
        try:
            vix_regime = compute_vix_regime(
                vix_close,
                low_threshold=cfg.vix_low_threshold,
                high_threshold=cfg.vix_high_threshold,
            )
            out.vix_level = partition_by_regime(result, vix_regime, axis_name="vix_level")
        except Exception as e:  # noqa: BLE001
            out.vix_level_error = f"{type(e).__name__}: {e}"
    elif vix_close is None:
        out.vix_level_error = "VIX data not provided"

    return out
