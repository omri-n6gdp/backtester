"""
Vectorized backtest core.

This is THE backtest. Every parameter sweep, every robustness analysis,
every dashboard chart eventually traces back to `run_backtest()`.

Critical invariant: positions are shifted by ONE BAR before being multiplied
by returns. This means a strategy that looks at bar t's close to decide
its position is, in effect, only allowed to act on that decision at bar t+1.
This makes look-ahead bias impossible by construction even when strategy
authors are careless.

The math:
    positions    = strategy(data, params)           # target on bar t
    positions    = positions.shift(1).fillna(0)     # applied at bar t+1
    returns      = data['close'].pct_change()
    costs        = compute_costs(positions, data, cost_config)
    strat_ret    = positions * returns - costs
    equity       = (1 + strat_ret).cumprod() * starting_capital
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .contract import POSITION_MAX, POSITION_MIN, REQUIRED_COLUMNS, StrategyFn
from .costs import DEFAULT_COST_CONFIG, CostConfig, compute_costs


@dataclass
class Trade:
    """A round-trip 'trade' in the simplified vectorized sense.

    Vectorized backtests don't have entry/exit fills the way an event-driven
    engine does. We define a 'trade' as a contiguous run during which the
    target position is non-zero and the sign is constant. Crossing through
    zero closes a trade and (if the next bar is non-zero) opens a new one.
    """
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # 'long' or 'short'
    avg_position: float
    entry_price: float
    exit_price: float
    bars_held: int
    pnl_return: float  # cumulative strategy return over the trade window


@dataclass
class BacktestResult:
    """Everything a single backtest produces. Plotted, persisted, analyzed downstream."""
    equity: pd.Series                       # equity curve
    drawdown: pd.Series                      # underwater curve (<= 0)
    returns: pd.Series                       # per-bar strategy returns (post-cost)
    positions: pd.Series                     # SHIFTED positions (the ones actually applied)
    raw_positions: pd.Series                 # UNSHIFTED positions (what the strategy returned)
    costs: pd.Series                         # per-bar costs in return units
    trades: list[Trade] = field(default_factory=list)
    starting_capital: float = 100_000.0
    bars_per_year: int = 252


# --- Public entry point ----------------------------------------------------

def run_backtest(
    strategy: StrategyFn,
    data: pd.DataFrame,
    params: Optional[dict] = None,
    *,
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
    starting_capital: float = 100_000.0,
    bars_per_year: int = 252,
) -> BacktestResult:
    """
    Run a vectorized backtest.

    Args:
        strategy: function (data, params) -> Series of target positions in [-1, +1].
        data:     OHLCV DataFrame.
        params:   strategy parameters (defaults to {}).
        cost_config: cost model config. See costs.py.
        starting_capital: initial equity.
        bars_per_year: 252 (daily), ~1638 (hourly equities), 252*78 (5m equities).

    Returns:
        BacktestResult.
    """
    _validate_data(data)
    p = params if params is not None else {}

    raw_positions = strategy(data, p)
    raw_positions = _coerce_positions(raw_positions, data.index)

    # === LOOK-AHEAD GUARD =================================================
    # A strategy decision computed at bar t can only be acted on at bar t+1.
    # Shifting by 1 here means a strategy that 'cheats' by referencing the
    # same bar's close is automatically penalized: its position only takes
    # effect on the *next* bar's return. Do NOT remove this shift. It is
    # tested by `test_look_ahead_bias_is_neutralized`.
    positions = raw_positions.shift(1).fillna(0.0)
    # ======================================================================

    returns = data["close"].pct_change().fillna(0.0)
    costs = compute_costs(positions, data, cost_config)
    strat_ret = positions * returns - costs

    equity = (1.0 + strat_ret).cumprod() * starting_capital
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0  # <= 0

    trades = _extract_trades(raw_positions, positions, data, strat_ret)

    return BacktestResult(
        equity=equity,
        drawdown=drawdown,
        returns=strat_ret,
        positions=positions,
        raw_positions=raw_positions,
        costs=costs,
        trades=trades,
        starting_capital=starting_capital,
        bars_per_year=bars_per_year,
    )


# --- Helpers ---------------------------------------------------------------

def _validate_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a DataFrame; got {type(data).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"data missing required columns: {missing}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data.index must be a DatetimeIndex")
    if len(data) < 2:
        raise ValueError(f"data has {len(data)} rows; need >= 2")


def _coerce_positions(positions: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex/clip positions to align with data and stay within bounds."""
    if not isinstance(positions, pd.Series):
        raise TypeError(f"strategy must return pd.Series; got {type(positions).__name__}")
    s = pd.to_numeric(positions, errors="coerce")
    if not s.index.equals(target_index):
        s = s.reindex(target_index)
    s = s.fillna(0.0).clip(lower=POSITION_MIN, upper=POSITION_MAX)
    return s.astype(float)


def _extract_trades(
    raw_positions: pd.Series,
    shifted_positions: pd.Series,
    data: pd.DataFrame,
    strat_ret: pd.Series,
) -> list[Trade]:
    """
    Extract round-trip trades by walking the shifted position series.

    A trade opens when position transitions from 0 (or sign flip) to non-zero,
    and closes when it returns to 0 (or sign flip). We use shifted positions
    because those are the ones actually 'in the market'.
    """
    trades: list[Trade] = []
    pos = shifted_positions.values
    idx = shifted_positions.index
    close = data["close"].values

    in_trade = False
    side = ""
    entry_i = 0
    pos_accum = 0.0
    pos_count = 0
    cum_ret_at_entry = 0.0
    cum_ret = (1.0 + strat_ret).cumprod().values

    for i in range(len(pos)):
        p = pos[i]
        if not in_trade:
            if p != 0:
                in_trade = True
                side = "long" if p > 0 else "short"
                entry_i = i
                pos_accum = p
                pos_count = 1
                cum_ret_at_entry = cum_ret[i - 1] if i > 0 else 1.0
        else:
            # Close conditions: position hits zero, or flips sign.
            sign_flip = (p > 0 and side == "short") or (p < 0 and side == "long")
            if p == 0 or sign_flip:
                exit_i = i  # trade exits at this bar's "open" conceptually
                trade_ret = cum_ret[exit_i] / cum_ret_at_entry - 1.0 if cum_ret_at_entry != 0 else 0.0
                trades.append(Trade(
                    entry_time=idx[entry_i],
                    exit_time=idx[exit_i],
                    side=side,
                    avg_position=pos_accum / max(pos_count, 1),
                    entry_price=float(close[entry_i]),
                    exit_price=float(close[exit_i]),
                    bars_held=exit_i - entry_i,
                    pnl_return=float(trade_ret),
                ))
                if sign_flip:
                    # Open new trade in the opposite direction immediately.
                    in_trade = True
                    side = "long" if p > 0 else "short"
                    entry_i = i
                    pos_accum = p
                    pos_count = 1
                    cum_ret_at_entry = cum_ret[i - 1] if i > 0 else 1.0
                else:
                    in_trade = False
            else:
                pos_accum += p
                pos_count += 1

    # Close any open trade at the end of the series.
    if in_trade:
        exit_i = len(pos) - 1
        trade_ret = cum_ret[exit_i] / cum_ret_at_entry - 1.0 if cum_ret_at_entry != 0 else 0.0
        trades.append(Trade(
            entry_time=idx[entry_i],
            exit_time=idx[exit_i],
            side=side,
            avg_position=pos_accum / max(pos_count, 1),
            entry_price=float(close[entry_i]),
            exit_price=float(close[exit_i]),
            bars_held=exit_i - entry_i,
            pnl_return=float(trade_ret),
        ))

    return trades
