"""
Metrics computed for every backtest.

Every function here is pure: takes a BacktestResult (or its components) and
returns a float or small dict. No side effects, no I/O. NumPy-only.

Grouped: return / risk / risk-adjusted / trade-level / benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .vectorized import BacktestResult, Trade


# --- Top-level -------------------------------------------------------------

@dataclass
class MetricsBundle:
    """All metrics for a single backtest. Persisted to SQLite as JSON."""
    # Return
    total_return: float
    cagr: float
    best_month: float
    worst_month: float

    # Risk
    annualized_vol: float
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    ulcer_index: float
    downside_deviation: float
    var_95: float
    var_99: float
    cvar_95: float

    # Risk-adjusted
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    mar: float

    # Trade-level
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    avg_holding_bars: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(result: BacktestResult) -> MetricsBundle:
    """One-shot computation of every metric for a BacktestResult."""
    bpy = result.bars_per_year
    r = result.returns

    return MetricsBundle(
        total_return=total_return(result.equity),
        cagr=cagr(result.equity, bpy),
        best_month=best_month(r),
        worst_month=worst_month(r),
        annualized_vol=annualized_vol(r, bpy),
        max_drawdown=max_drawdown(result.drawdown),
        max_drawdown_duration_days=max_drawdown_duration_bars(result.equity),
        avg_drawdown=avg_drawdown(result.drawdown),
        ulcer_index=ulcer_index(result.drawdown),
        downside_deviation=downside_deviation(r, bpy),
        var_95=value_at_risk(r, 0.95),
        var_99=value_at_risk(r, 0.99),
        cvar_95=conditional_var(r, 0.95),
        sharpe=sharpe(r, bpy),
        sortino=sortino(r, bpy),
        calmar=calmar(result.equity, result.drawdown, bpy),
        omega=omega(r),
        mar=mar(result.equity, result.drawdown, bpy),
        n_trades=len(result.trades),
        win_rate=win_rate(result.trades),
        profit_factor=profit_factor(result.trades),
        expectancy=expectancy(result.trades),
        avg_win=avg_win(result.trades),
        avg_loss=avg_loss(result.trades),
        avg_holding_bars=avg_holding_bars(result.trades),
        max_consecutive_wins=max_consecutive(result.trades, win=True),
        max_consecutive_losses=max_consecutive(result.trades, win=False),
    )


# --- Return ----------------------------------------------------------------

def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, bars_per_year: int) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    n_bars = len(equity) - 1
    years = n_bars / bars_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def best_month(returns: pd.Series) -> float:
    monthly = (1 + returns).resample("ME").prod() - 1
    return float(monthly.max()) if len(monthly) else 0.0


def worst_month(returns: pd.Series) -> float:
    monthly = (1 + returns).resample("ME").prod() - 1
    return float(monthly.min()) if len(monthly) else 0.0


# --- Risk ------------------------------------------------------------------

def annualized_vol(returns: pd.Series, bars_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(bars_per_year))


def max_drawdown(drawdown: pd.Series) -> float:
    """Most negative drawdown value. Returned as a negative number (e.g. -0.25)."""
    if len(drawdown) == 0:
        return 0.0
    return float(drawdown.min())


def max_drawdown_duration_bars(equity: pd.Series) -> int:
    """Longest run of bars where equity is below its prior peak."""
    if len(equity) == 0:
        return 0
    peak = equity.cummax()
    underwater = equity < peak
    if not underwater.any():
        return 0
    # Run-length encoding via groupby on cumulative resets at peak-touches.
    grp = (~underwater).cumsum()
    runs = underwater.groupby(grp).sum()
    return int(runs.max())


def avg_drawdown(drawdown: pd.Series) -> float:
    """Average depth of drawdown across underwater periods only."""
    underwater = drawdown[drawdown < 0]
    return float(underwater.mean()) if len(underwater) else 0.0


def ulcer_index(drawdown: pd.Series) -> float:
    """sqrt(mean(drawdown^2)). A lower-is-better DD-magnitude measure."""
    if len(drawdown) == 0:
        return 0.0
    return float(np.sqrt((drawdown ** 2).mean()))


def downside_deviation(returns: pd.Series, bars_per_year: int, mar_rate: float = 0.0) -> float:
    """Annualized stdev of returns BELOW the MAR. Used in Sortino."""
    excess = returns - mar_rate / bars_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    return float(downside.std(ddof=1) * np.sqrt(bars_per_year))


def value_at_risk(returns: pd.Series, confidence: float) -> float:
    """Historical VaR. Returned as a negative number (the loss threshold)."""
    if len(returns) == 0:
        return 0.0
    return float(np.quantile(returns, 1 - confidence))


def conditional_var(returns: pd.Series, confidence: float) -> float:
    """Expected loss conditional on being in the worst (1-confidence) tail."""
    if len(returns) == 0:
        return 0.0
    threshold = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) else float(threshold)


# --- Risk-adjusted ---------------------------------------------------------

def sharpe(returns: pd.Series, bars_per_year: int, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / bars_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(bars_per_year))


def sortino(returns: pd.Series, bars_per_year: int, mar_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - mar_rate / bars_per_year
    dd = downside_deviation(returns, bars_per_year, mar_rate)
    if dd == 0:
        return 0.0
    return float(excess.mean() * bars_per_year / dd)


def calmar(equity: pd.Series, drawdown: pd.Series, bars_per_year: int) -> float:
    mdd = abs(max_drawdown(drawdown))
    if mdd == 0:
        return 0.0
    return float(cagr(equity, bars_per_year) / mdd)


def omega(returns: pd.Series, threshold: float = 0.0) -> float:
    """Probability-weighted gains-over-losses ratio."""
    if len(returns) == 0:
        return 0.0
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def mar(equity: pd.Series, drawdown: pd.Series, bars_per_year: int) -> float:
    """MAR = CAGR / |max DD|. Same shape as Calmar; provided for completeness."""
    return calmar(equity, drawdown, bars_per_year)


# --- Trade-level -----------------------------------------------------------

def win_rate(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl_return > 0)
    return wins / len(trades)


def profit_factor(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    gross_win = sum(t.pnl_return for t in trades if t.pnl_return > 0)
    gross_loss = -sum(t.pnl_return for t in trades if t.pnl_return < 0)
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t.pnl_return for t in trades]))


def avg_win(trades: list[Trade]) -> float:
    wins = [t.pnl_return for t in trades if t.pnl_return > 0]
    return float(np.mean(wins)) if wins else 0.0


def avg_loss(trades: list[Trade]) -> float:
    losses = [t.pnl_return for t in trades if t.pnl_return < 0]
    return float(np.mean(losses)) if losses else 0.0


def avg_holding_bars(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t.bars_held for t in trades]))


def max_consecutive(trades: list[Trade], *, win: bool) -> int:
    """Longest streak of wins (win=True) or losses (win=False)."""
    if not trades:
        return 0
    best = cur = 0
    for t in trades:
        is_win = t.pnl_return > 0
        if is_win == win:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
