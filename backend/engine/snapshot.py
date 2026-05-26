"""
Phase 17.3 — as-of-date metric snapshots.

"What would this strategy's metrics have looked like at the end of each
calendar year?" Helpful for spotting whether the edge is recent or
robust across decades.

Pure compute over a stored equity / returns / drawdown trio (which is
what `runs.curves` carries). Returns one snapshot per calendar year
covered by the data, each computed from the strategy's start through
the last bar of that year. The final partial year is included if the
data ends before December.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from backend.engine.metrics import (
    cagr as _cagr,
    max_drawdown as _max_drawdown,
    sharpe as _sharpe,
    total_return as _total_return,
)


@dataclass(frozen=True)
class YearSnapshot:
    """One year's view of the strategy's cumulative metrics."""
    year: int
    year_end: str            # ISO date of the last bar within the year
    n_bars: int              # bars from start through year_end
    total_return_to_date: float
    cagr_to_date: float
    sharpe_to_date: float
    max_drawdown_to_date: float
    win_rate_to_date: float


def _drawdown_from_equity(equity: pd.Series) -> pd.Series:
    """Drawdown as a negative fraction. Reproduces what the vectorized
    engine stores so we don't need the original `BacktestResult` here."""
    if len(equity) == 0:
        return equity.copy()
    peak = equity.cummax()
    return (equity / peak) - 1.0


def compute_yearly_snapshots(
    timestamps: Iterable[str],
    equity: Iterable[float | None],
    returns: Iterable[float | None],
    bars_per_year: int = 252,
) -> list[YearSnapshot]:
    """Build one YearSnapshot per calendar year of the run."""
    idx = pd.to_datetime(list(timestamps), utc=True)
    if len(idx) == 0:
        return []
    eq = pd.Series([None if v is None else float(v) for v in equity], index=idx).dropna()
    rs = pd.Series([None if v is None else float(v) for v in returns], index=idx).dropna()
    if eq.empty:
        return []

    snapshots: list[YearSnapshot] = []
    years = sorted(set(eq.index.year))
    for year in years:
        year_end_ts = eq.index[eq.index.year == year][-1]
        eq_slice = eq.loc[:year_end_ts]
        rs_slice = rs.loc[:year_end_ts] if not rs.empty else rs
        if len(eq_slice) < 2:
            continue
        dd_slice = _drawdown_from_equity(eq_slice)

        # win_rate expects a trades list; we don't have that level of
        # detail in the slice, so use a per-bar approximation: fraction
        # of positive return bars. Documented in the field name (it's
        # _win_rate_to_date, the bar-level approximation, not the
        # trade-level metric).
        if len(rs_slice) > 0:
            wins = (rs_slice > 0).sum()
            total = (rs_slice != 0).sum()
            wr = float(wins / total) if total > 0 else 0.0
        else:
            wr = 0.0

        snapshots.append(YearSnapshot(
            year=int(year),
            year_end=year_end_ts.isoformat(),
            n_bars=int(len(eq_slice)),
            total_return_to_date=_total_return(eq_slice),
            cagr_to_date=_cagr(eq_slice, bars_per_year),
            sharpe_to_date=_sharpe(rs_slice, bars_per_year),
            max_drawdown_to_date=_max_drawdown(dd_slice),
            win_rate_to_date=wr,
        ))
    return snapshots
