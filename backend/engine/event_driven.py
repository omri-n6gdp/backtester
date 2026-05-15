"""
Event-driven backtest mode (Phase 9).

The vectorized engine treats positions as held bar-to-bar. That can't
express intra-bar stops (stop-loss, take-profit, trailing stop) which
trigger mid-bar based on bar.high / bar.low. This module fixes that.

Contract: same `strategy(data, params) -> pd.Series` as vectorized. The
strategy still returns target positions; this engine walks bar-by-bar
to apply protective stops on top. The output is a `BacktestResult` of
the same shape, so the entire downstream stack (metrics, OOS, MC,
regimes, library) works unchanged.

Semantics
---------
- Strategy positions are shifted by 1 (same look-ahead guard as vectorized).
- For each bar: update peak/trough (bar.high / bar.low) if a position is open,
  then check stops in priority order: stop_loss -> take_profit -> trailing.
- A stop fires at the trigger price, but if the bar gapped past it
  (bar.open ≤ stop on a long), fill at the open — worst case for the
  holder.
- When a stop fires on bar t, the position is closed mid-bar (actual
  position for bar t is 0). If the strategy's signal still wants the
  position on bar t+1, we re-open at bar t's close. The "no immediate
  re-entry" only applies WITHIN the bar that the stop fired in.
- The bar return on a stop-hit bar is *partial*: we held the position
  from prev_close to the stop price, then flat for the rest. That
  partial return is what reaches the equity curve, costs are taken
  on the close transaction (size = abs(position)).
- Trigger comparisons use a small epsilon (`_PRICE_EPS`) to handle
  floating-point imprecision in price-times-percent math (e.g.
  `100.0 * 1.10 != 110.0` exactly), so a stop at exactly the
  threshold fires deterministically.

Cost model: same `compute_costs` as vectorized — slippage / commission /
spread / FX / borrow are functions of position changes and aggregate
exposure. Trade costs are charged on the closing bar in the same units.

Look-ahead protection: inherited from the `.shift(1)` on the target
positions. Stop triggers use bar.high / bar.low / bar.open, all of
which are part of the bar being acted on (not future bars).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .contract import StrategyFn
from .costs import DEFAULT_COST_CONFIG, CostConfig, compute_costs
from .vectorized import BacktestResult, Trade, _coerce_positions, _validate_data


@dataclass(frozen=True)
class StopConfig:
    """Protective stops attached to a position.

    All percents are non-negative fractions: 0.05 = 5%. None means "no
    stop of this kind". Multiple stops can be active simultaneously;
    they fire in priority order stop_loss -> take_profit -> trailing.
    """
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None

    def __post_init__(self) -> None:
        for name, val in (
            ("stop_loss_pct", self.stop_loss_pct),
            ("take_profit_pct", self.take_profit_pct),
            ("trailing_stop_pct", self.trailing_stop_pct),
        ):
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                raise ValueError(f"{name} must be a number; got {type(val).__name__}")
            if val < 0 or val >= 1:
                raise ValueError(f"{name} must be in [0, 1); got {val}")

    def any_active(self) -> bool:
        return any(p is not None for p in (
            self.stop_loss_pct, self.take_profit_pct, self.trailing_stop_pct,
        ))


# Constants reused for trade exit reasons. Strings so they round-trip
# through JSON cleanly.
_REASON_STOP_LOSS = "stop_loss"
_REASON_TAKE_PROFIT = "take_profit"
_REASON_TRAILING = "trailing_stop"
_REASON_SIGNAL = "signal"

# Float epsilon for stop threshold comparisons. Prices times percents
# can land just past representable doubles (100.0 * 1.10 evaluates to
# 110.00000000000001); without this epsilon, a stop at exactly the
# threshold would silently fail to trigger.
_PRICE_EPS = 1e-9


def run_event_driven(
    strategy: StrategyFn,
    data: pd.DataFrame,
    params: Optional[dict] = None,
    *,
    stop_config: StopConfig = StopConfig(),
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
    starting_capital: float = 100_000.0,
    bars_per_year: int = 252,
) -> BacktestResult:
    """Event-driven backtest with protective stops.

    When `stop_config` has no active stops, the result is mathematically
    identical to `run_backtest` (a regression test enforces this).
    """
    _validate_data(data)
    p = params if params is not None else {}

    raw_positions = strategy(data, p)
    raw_positions = _coerce_positions(raw_positions, data.index)

    # Same look-ahead guard as vectorized: positions[t] is what we hold
    # going INTO bar t (i.e., the signal at bar t-1).
    targets = raw_positions.shift(1).fillna(0.0).to_numpy(dtype=float)

    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    n = len(data)

    actual_positions = np.zeros(n, dtype=float)
    bar_returns = np.zeros(n, dtype=float)  # per-bar strategy return BEFORE costs

    # Trade-tracking state:
    current_pos = 0.0
    entry_price = 0.0
    entry_i = -1
    peak_high = 0.0      # highest high since entry (for long trailing)
    trough_low = 0.0     # lowest low since entry (for short trailing)
    just_closed_this_bar = False

    trades: list[Trade] = []

    SL = stop_config.stop_loss_pct
    TP = stop_config.take_profit_pct
    TS = stop_config.trailing_stop_pct

    for i in range(n):
        target = float(targets[i])
        prev_close = closes[i - 1] if i > 0 else closes[i]
        bar_open, bar_high, bar_low, bar_close = (
            opens[i], highs[i], lows[i], closes[i],
        )

        # 1. Maintain trailing extrema for any open position.
        if current_pos > 0:
            if bar_high > peak_high:
                peak_high = bar_high
        elif current_pos < 0:
            if bar_low < trough_low or trough_low == 0.0:
                trough_low = bar_low

        # 2. Check for stop triggers (only if we have an open position).
        exit_reason: Optional[str] = None
        exit_price: float = 0.0

        if current_pos > 0:
            # LONG: SL/TS trigger when low <= trigger; TP triggers when high >= trigger.
            triggers: list[tuple[float, str]] = []
            if SL is not None:
                trig = entry_price * (1.0 - SL)
                if bar_low <= trig + _PRICE_EPS:
                    fill = min(trig, bar_open) if bar_open < trig else trig
                    triggers.append((fill, _REASON_STOP_LOSS))
            if TP is not None:
                trig = entry_price * (1.0 + TP)
                if bar_high >= trig - _PRICE_EPS:
                    fill = max(trig, bar_open) if bar_open > trig else trig
                    triggers.append((fill, _REASON_TAKE_PROFIT))
            if TS is not None:
                trig = peak_high * (1.0 - TS)
                if bar_low <= trig + _PRICE_EPS:
                    fill = min(trig, bar_open) if bar_open < trig else trig
                    triggers.append((fill, _REASON_TRAILING))
            # Priority order matches list order above (SL -> TP -> TS).
            if triggers:
                exit_price, exit_reason = triggers[0]
        elif current_pos < 0:
            # SHORT: mirror — SL/TS trigger when high >= trigger; TP triggers when low <= trigger.
            triggers = []
            if SL is not None:
                trig = entry_price * (1.0 + SL)
                if bar_high >= trig - _PRICE_EPS:
                    fill = max(trig, bar_open) if bar_open > trig else trig
                    triggers.append((fill, _REASON_STOP_LOSS))
            if TP is not None:
                trig = entry_price * (1.0 - TP)
                if bar_low <= trig + _PRICE_EPS:
                    fill = min(trig, bar_open) if bar_open < trig else trig
                    triggers.append((fill, _REASON_TAKE_PROFIT))
            if TS is not None:
                trig = trough_low * (1.0 + TS)
                if bar_high >= trig - _PRICE_EPS:
                    fill = max(trig, bar_open) if bar_open > trig else trig
                    triggers.append((fill, _REASON_TRAILING))
            if triggers:
                exit_price, exit_reason = triggers[0]

        just_closed_this_bar = False
        if exit_reason is not None and current_pos != 0:
            # Stop fired. Compute partial bar return: held the position
            # from prev_close to exit_price, then flat.
            partial = (exit_price - prev_close) / prev_close if prev_close != 0 else 0.0
            bar_returns[i] = current_pos * partial
            actual_positions[i] = 0.0  # closed during this bar

            trades.append(Trade(
                entry_time=data.index[entry_i],
                exit_time=data.index[i],
                side="long" if current_pos > 0 else "short",
                avg_position=current_pos,
                entry_price=entry_price,
                exit_price=exit_price,
                bars_held=i - entry_i,
                pnl_return=(exit_price / entry_price - 1.0) * (1.0 if current_pos > 0 else -1.0),
                exit_reason=exit_reason,
            ))
            current_pos = 0.0
            entry_price = 0.0
            entry_i = -1
            peak_high = 0.0
            trough_low = 0.0
            just_closed_this_bar = True
        else:
            # No stop fired this bar. The position is held for the full bar.
            ret = (bar_close - prev_close) / prev_close if prev_close != 0 else 0.0
            bar_returns[i] = current_pos * ret
            actual_positions[i] = current_pos

        # 3. Honor target changes. We update current_pos AFTER computing
        # this bar's return, so any change takes effect on the NEXT bar's
        # hold — preserving the same look-ahead semantics as vectorized.
        # If a stop fired this bar, current_pos was just reset to 0; the
        # signal-change branch below can still re-open if the target is
        # non-zero (the position then holds from the NEXT bar onward —
        # bar t's `actual_positions[t]` is already 0).
        if i + 1 < n:
            next_target = float(targets[i + 1])
            if next_target != current_pos:
                # Close any existing position at this bar's close (signal exit).
                if current_pos != 0.0:
                    trades.append(Trade(
                        entry_time=data.index[entry_i],
                        exit_time=data.index[i],
                        side="long" if current_pos > 0 else "short",
                        avg_position=current_pos,
                        entry_price=entry_price,
                        exit_price=bar_close,
                        bars_held=i - entry_i,
                        pnl_return=(bar_close / entry_price - 1.0) * (1.0 if current_pos > 0 else -1.0),
                        exit_reason=_REASON_SIGNAL,
                    ))
                # Open new position (or flat) at this bar's close.
                current_pos = next_target
                if next_target != 0.0:
                    entry_price = bar_close
                    entry_i = i
                    peak_high = bar_high if next_target > 0 else 0.0
                    trough_low = bar_low if next_target < 0 else 0.0
                else:
                    entry_price = 0.0
                    entry_i = -1
                    peak_high = 0.0
                    trough_low = 0.0

    # Close any still-open trade at the final bar's close.
    if current_pos != 0.0:
        trades.append(Trade(
            entry_time=data.index[entry_i],
            exit_time=data.index[-1],
            side="long" if current_pos > 0 else "short",
            avg_position=current_pos,
            entry_price=entry_price,
            exit_price=closes[-1],
            bars_held=(n - 1) - entry_i,
            pnl_return=(closes[-1] / entry_price - 1.0) * (1.0 if current_pos > 0 else -1.0),
            exit_reason=_REASON_SIGNAL,
        ))

    positions_s = pd.Series(actual_positions, index=data.index)
    costs = compute_costs(positions_s, data, cost_config)
    bar_ret_s = pd.Series(bar_returns, index=data.index)
    strat_ret = bar_ret_s - costs

    equity = (1.0 + strat_ret).cumprod() * starting_capital
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0

    return BacktestResult(
        equity=equity,
        drawdown=drawdown,
        returns=strat_ret,
        positions=positions_s,
        raw_positions=raw_positions,
        costs=costs,
        trades=trades,
        starting_capital=starting_capital,
        bars_per_year=bars_per_year,
    )
