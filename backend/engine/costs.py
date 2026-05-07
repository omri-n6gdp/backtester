"""
Cost model for backtests.

Costs are applied per bar, proportional to the *change* in target position
(turnover). Each component is independently configurable. The defaults
match Trading 212 reality for a UK retail user trading non-GBP equities:
no commission, modest slippage, 15 bps FX fee.

Per-bar cost in absolute terms (i.e. directly subtracted from per-bar return):

    turnover[t] = abs(positions[t] - positions[t-1])
    cost[t]     = turnover[t] * (slippage + commission + spread + fx) / 10_000
                + abs(positions[t]) * borrow_cost_per_bar  (only when negative)

Slippage can also be ATR-scaled (vol_scaled mode); see compute_costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    """
    Cost configuration. All fields in basis points unless noted.

    Defaults: 5 bps slippage, 0 bps commission, 0 bps spread (folded into slippage),
    15 bps FX fee (auto-zero if instrument is GBP-denominated), 0 bps borrow.
    """

    # Slippage in bps (per-side). Applied to abs change in position.
    slippage_bps: float = 5.0

    # Slippage mode: 'fixed' uses slippage_bps directly; 'vol_scaled' multiplies
    # by 0.5 * (rolling 14-bar ATR%) so volatile bars cost more.
    slippage_mode: str = "fixed"  # 'fixed' | 'vol_scaled'

    # Commission per trade, in bps of notional (T212 = 0).
    commission_bps: float = 0.0

    # Bid-ask spread cost in bps. Often folded into slippage; default 0.
    spread_bps: float = 0.0

    # FX fee in bps for non-GBP instruments. 0 if instrument is GBP.
    fx_bps: float = 15.0

    # Whether the instrument is GBP-denominated. If True, fx_bps is ignored.
    instrument_is_gbp: bool = False

    # Daily borrow cost in bps for short positions (annualized / 252 internally).
    # Applied only when position < 0, proportional to abs(position).
    borrow_bps_annual: float = 0.0

    # Bars per year for annualization (252 daily, ~6552 hourly, ~78624 5m).
    bars_per_year: int = 252


# Defaults exposed as a module constant for callers that don't want to think.
DEFAULT_COST_CONFIG = CostConfig()


def compute_costs(
    positions: pd.Series,
    data: pd.DataFrame,
    config: CostConfig = DEFAULT_COST_CONFIG,
) -> pd.Series:
    """
    Compute per-bar costs as a fraction of capital (subtract directly from returns).

    Args:
        positions: target positions AFTER any look-ahead shift (engine handles shift;
                   this function does not shift).
        data:      OHLCV DataFrame aligned with `positions`.
        config:    cost configuration.

    Returns:
        Series of per-bar costs in return units (e.g. 0.0005 = 5 bps).
    """
    if positions.empty:
        return pd.Series(dtype=float, index=positions.index)

    # Turnover: |Δposition| per bar. First bar's turnover = |first position|.
    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0]) if len(positions) > 0 else 0.0

    # Per-trade bps total (slippage + commission + spread + FX).
    fx = 0.0 if config.instrument_is_gbp else config.fx_bps

    if config.slippage_mode == "fixed":
        slippage = pd.Series(config.slippage_bps, index=positions.index)
    elif config.slippage_mode == "vol_scaled":
        # ATR%: 14-bar Wilder-style ATR / close.
        atr_pct = _atr_percent(data, window=14)
        slippage = 0.5 * atr_pct * 10_000  # convert fraction back to bps
        slippage = slippage.fillna(config.slippage_bps).clip(lower=0)
    else:
        raise ValueError(f"Unknown slippage_mode: {config.slippage_mode!r}")

    per_trade_bps = slippage + config.commission_bps + config.spread_bps + fx
    trade_cost = turnover * per_trade_bps / 10_000.0

    # Borrow cost: applied only on short side, proportional to abs(position).
    borrow_per_bar = config.borrow_bps_annual / 10_000.0 / config.bars_per_year
    short_exposure = positions.where(positions < 0, 0.0).abs()
    borrow_cost = short_exposure * borrow_per_bar

    return trade_cost.add(borrow_cost, fill_value=0.0)


def _atr_percent(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range as a fraction of close. Wilder's smoothing approximated
    with simple EMA for simplicity (close enough for cost modeling).
    """
    high = data["high"]
    low = data["low"]
    prev_close = data["close"].shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return atr / data["close"]
