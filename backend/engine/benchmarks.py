"""
Phase 17.2 — configurable benchmark suite.

A benchmark is a passive portfolio whose equity curve gets overlaid on
a backtest. Each benchmark is described by an ID, a human label, and a
function that returns an `equity` Series indexed by the bars the user
asked for.

Free, yfinance-backed only:
  - spy_bh        Buy & hold SPY
  - qqq_bh        Buy & hold QQQ
  - gold          Buy & hold GLD
  - sixty_forty   60% SPY + 40% AGG, monthly rebalance
  - mag7_eq       Equal-weight Mag-7 (AAPL/MSFT/AMZN/NVDA/GOOGL/META/TSLA),
                  monthly rebalance

The route handler calls `compute_benchmarks(names, start, end, capital,
target_index)`; per-benchmark errors are captured so a single bad
ticker (or a rate-limit hiccup) never fails the whole backtest. The
caller may pass `target_index` (the equity-curve index the user wants
the benchmark to align to); the benchmark series is reindexed +
forward-filled to that index before return.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from backend.data.prices import get_prices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkDef:
    id: str
    label: str
    description: str
    # tickers used. Empty for derived strategies that pull more than
    # one inside their compute func.
    tickers: tuple[str, ...]


def _buy_hold(ticker: str) -> Callable:
    def compute(start: date, end: date, capital: float) -> pd.Series:
        df = get_prices(ticker=ticker, timeframe="1d", start=start, end=end)
        if df is None or df.empty:
            raise RuntimeError(f"no data for {ticker}")
        close = df["close"].dropna()
        if close.empty:
            raise RuntimeError(f"empty close column for {ticker}")
        return capital * close / close.iloc[0]
    return compute


def _equal_weight_basket(tickers: tuple[str, ...]) -> Callable:
    """Equal-weight basket, rebalanced monthly. Each constituent's
    Buy & Hold curve is normalized to its weight; we sum the curves
    on each bar to get the portfolio equity (continuous rebalancing is
    a good approximation for monthly when bars are daily — and the user
    is comparing shapes, not exact returns)."""
    def compute(start: date, end: date, capital: float) -> pd.Series:
        curves: list[pd.Series] = []
        for tk in tickers:
            try:
                df = get_prices(ticker=tk, timeframe="1d", start=start, end=end)
                if df is None or df.empty:
                    continue
                close = df["close"].dropna()
                if close.empty:
                    continue
                curves.append(close / close.iloc[0])
            except Exception:  # noqa: BLE001 — one bad ticker shouldn't fail the basket
                logger.exception("benchmark basket: %s fetch failed", tk)
        if not curves:
            raise RuntimeError(f"no constituents fetched for {tickers}")
        aligned = pd.concat(curves, axis=1).ffill()
        # Equal-weight at each bar; gives a continuously rebalanced
        # portfolio. Drop bars where every constituent is NaN.
        mean = aligned.mean(axis=1).dropna()
        return capital * mean
    return compute


def _sixty_forty() -> Callable:
    """60% SPY + 40% AGG, daily rebalance (close to monthly under
    realistic conditions; the user is comparing shapes)."""
    def compute(start: date, end: date, capital: float) -> pd.Series:
        spy = get_prices(ticker="SPY", timeframe="1d", start=start, end=end)
        agg = get_prices(ticker="AGG", timeframe="1d", start=start, end=end)
        if spy is None or spy.empty:
            raise RuntimeError("no data for SPY")
        if agg is None or agg.empty:
            raise RuntimeError("no data for AGG")
        spy_c = spy["close"].dropna()
        agg_c = agg["close"].dropna()
        df = pd.concat({"SPY": spy_c, "AGG": agg_c}, axis=1).ffill().dropna()
        weighted = 0.6 * (df["SPY"] / df["SPY"].iloc[0]) + 0.4 * (df["AGG"] / df["AGG"].iloc[0])
        return capital * weighted
    return compute


# The static benchmark catalogue. New entries are additive.
BENCHMARKS: dict[str, tuple[BenchmarkDef, Callable]] = {
    "spy_bh": (
        BenchmarkDef("spy_bh", "SPY buy & hold", "S&P 500 ETF, buy at start.",
                     ("SPY",)),
        _buy_hold("SPY"),
    ),
    "qqq_bh": (
        BenchmarkDef("qqq_bh", "QQQ buy & hold",
                     "Nasdaq-100 ETF, buy at start.",
                     ("QQQ",)),
        _buy_hold("QQQ"),
    ),
    "gold": (
        BenchmarkDef("gold", "Gold (GLD)",
                     "Gold ETF, buy at start.",
                     ("GLD",)),
        _buy_hold("GLD"),
    ),
    "sixty_forty": (
        BenchmarkDef("sixty_forty", "60 / 40 (SPY+AGG)",
                     "60% S&P 500 + 40% US aggregate bonds, continuous rebalance.",
                     ("SPY", "AGG")),
        _sixty_forty(),
    ),
    "mag7_eq": (
        BenchmarkDef("mag7_eq", "Magnificent 7 (equal-weight)",
                     "AAPL/MSFT/AMZN/NVDA/GOOGL/META/TSLA, continuous rebalance.",
                     ("AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA")),
        _equal_weight_basket(("AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA")),
    ),
}


def list_benchmarks() -> list[BenchmarkDef]:
    """Static catalogue exposed to the frontend so it can populate the
    selector without hard-coding labels."""
    return [defn for defn, _fn in BENCHMARKS.values()]


def compute_benchmarks(
    names: list[str],
    start: date,
    end: date,
    capital: float,
    target_index: pd.DatetimeIndex | None = None,
) -> dict[str, dict]:
    """Compute the requested benchmark equity curves.

    Returns a dict keyed by benchmark id. Each value is
    `{label, timestamps: [iso], equity: [float|None], error: str|None}`.
    Errors are captured per-benchmark so one bad ticker never fails the
    enclosing backtest.

    `target_index` (typically the strategy's equity index) lets the
    chart overlay align bar-for-bar; missing values come back as None.
    """
    out: dict[str, dict] = {}
    for name in names:
        entry = BENCHMARKS.get(name)
        if entry is None:
            out[name] = {
                "label": name,
                "timestamps": [],
                "equity": [],
                "error": f"unknown benchmark '{name}'",
            }
            continue
        defn, compute_fn = entry
        try:
            series = compute_fn(start, end, capital)
            if target_index is not None:
                series = series.reindex(target_index).ffill()
            timestamps = [t.isoformat() for t in series.index]
            equity = [
                None if pd.isna(v) else float(v)
                for v in series.to_list()
            ]
            out[name] = {
                "label": defn.label,
                "timestamps": timestamps,
                "equity": equity,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.exception("benchmark %s failed", name)
            out[name] = {
                "label": defn.label,
                "timestamps": [],
                "equity": [],
                "error": f"{type(e).__name__}: {e!s}",
            }
    return out
