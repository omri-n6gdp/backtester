"""
Unit tests for backend/engine/event_driven.py (Phase 9).

Most tests use hand-built OHLCV with known shapes so we can assert
exact stop fill prices and exit reasons.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.engine.costs import CostConfig
from backend.engine.event_driven import StopConfig, run_event_driven
from backend.engine.vectorized import run_backtest


# Zero costs so we can assert raw price math without subtracting noise.
ZERO_COSTS = CostConfig(
    slippage_bps=0.0, commission_bps=0.0, spread_bps=0.0,
    fx_bps=0.0, instrument_is_gbp=True, borrow_bps_annual=0.0,
)


def _ohlcv_from_close(close_values, *, open_offset=0.0, high_offset=0.0, low_offset=0.0):
    """Build a synthetic OHLCV frame from a list of closes.

    `open_offset` / `high_offset` / `low_offset` are added to the corresponding
    column (defaults make OHLC all equal to close — a constant-price bar). Use
    these to construct specific intra-bar shapes per row by passing iterables.
    """
    n = len(close_values)
    close = np.asarray(close_values, dtype=float)
    if np.isscalar(open_offset):
        open_arr = close + open_offset
    else:
        open_arr = close + np.asarray(open_offset, dtype=float)
    if np.isscalar(high_offset):
        high = close + high_offset
    else:
        high = close + np.asarray(high_offset, dtype=float)
    if np.isscalar(low_offset):
        low = close + low_offset
    else:
        low = close + np.asarray(low_offset, dtype=float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "open": open_arr,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1e6),
    }, index=idx)


# A strategy that goes long on bar 0 and stays long forever. Its position
# at bar t is 1.0 for t >= 0. After the engine's .shift(1), this becomes
# "hold from bar 1 onward".
def long_and_hold(data, params):
    return pd.Series(1.0, index=data.index)


# Same idea but short.
def short_and_hold(data, params):
    return pd.Series(-1.0, index=data.index)


# ---- stop-loss ---------------------------------------------------------

def test_stop_loss_triggers_at_threshold_for_long():
    # Closes: 100, 100, 100, 100, 94 (-6%). On bar 4 the low touches 94.
    # SL = 5% means stop at 95. The bar's low (94) <= 95, so SL fires.
    # Bar's open is 94 too (gap-down past stop), so fill at open = 94.
    df = _ohlcv_from_close([100, 100, 100, 100, 94])
    result = run_event_driven(
        long_and_hold, df, {},
        stop_config=StopConfig(stop_loss_pct=0.05),
        cost_config=ZERO_COSTS,
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == 94.0  # filled at gap-down open
    assert t.entry_price == 100.0


def test_stop_loss_filled_at_trigger_when_not_gapped():
    # Bar 4 opens at 100 then dips to 94 intra-bar — bar.open > stop > bar.low.
    # Fill at the stop price (95.0), not at the worse 94.
    closes = [100, 100, 100, 100, 100]  # final close unused (stop fires)
    df = _ohlcv_from_close(
        closes,
        open_offset=[0, 0, 0, 0, 0],     # open == close
        high_offset=[0, 0, 0, 0, 0],
        low_offset=[0, 0, 0, 0, -6],     # bar 4 low = 94
    )
    result = run_event_driven(
        long_and_hold, df, {},
        stop_config=StopConfig(stop_loss_pct=0.05),
        cost_config=ZERO_COSTS,
    )
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(95.0)


# ---- take-profit -------------------------------------------------------

def test_take_profit_triggers_for_long():
    # Closes climb past +10%. Bar 4 high reaches 110.
    df = _ohlcv_from_close([100, 102, 104, 108, 110])
    result = run_event_driven(
        long_and_hold, df, {},
        stop_config=StopConfig(take_profit_pct=0.10),
        cost_config=ZERO_COSTS,
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "take_profit"
    # Bar 4 has open=high=close=110. Fill at the trigger price (open == trig).
    assert t.exit_price == pytest.approx(110.0)


# ---- trailing stop -----------------------------------------------------

def test_trailing_stop_tracks_peak_and_triggers():
    # Price climbs to 120 then drops to 113. 5% trail from peak 120 = 114.
    # On the dropping bar, low touches 113 < 114, so trailing stop fires.
    closes = [100, 105, 110, 120, 113]
    df = _ohlcv_from_close(closes)
    result = run_event_driven(
        long_and_hold, df, {},
        stop_config=StopConfig(trailing_stop_pct=0.05),
        cost_config=ZERO_COSTS,
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "trailing_stop"
    # Bar 4 open == close == 113 (gap-down below trail 114).
    assert t.exit_price == 113.0


# ---- priority + symmetry ----------------------------------------------

def test_priority_stop_loss_beats_take_profit_in_same_bar():
    # Construct a bar where low <= SL AND high >= TP. SL should win.
    closes = [100, 100]
    df = _ohlcv_from_close(
        [100, 100],
        open_offset=[0, 0],
        high_offset=[0, 12],   # high = 112 — TP would fire at 110
        low_offset=[0, -7],    # low = 93 — SL would fire at 95
    )
    result = run_event_driven(
        long_and_hold, df, {},
        stop_config=StopConfig(stop_loss_pct=0.05, take_profit_pct=0.10),
        cost_config=ZERO_COSTS,
    )
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"


def test_short_position_stop_loss_triggers_on_high():
    # Short entry at 100. SL=5% means stop at 105. Bar 3 high = 106.
    closes = [100, 100, 100, 100]
    df = _ohlcv_from_close(
        closes, high_offset=[0, 0, 0, 6], open_offset=[0, 0, 0, 0], low_offset=[0, 0, 0, 0],
    )
    result = run_event_driven(
        short_and_hold, df, {},
        stop_config=StopConfig(stop_loss_pct=0.05),
        cost_config=ZERO_COSTS,
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.side == "short"
    assert t.exit_reason == "stop_loss"
    # Open == close == 100, high = 106. open (100) < stop (105), so fill at stop.
    assert t.exit_price == pytest.approx(105.0)


# ---- regression: no stops -> identical to vectorized -------------------

def test_no_stops_matches_vectorized_engine():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.012, 200)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1e6,
    }, index=idx)

    vec = run_backtest(long_and_hold, df, {}, cost_config=ZERO_COSTS)
    ev = run_event_driven(long_and_hold, df, {}, stop_config=StopConfig(), cost_config=ZERO_COSTS)

    # Final equity should match within numerical tolerance.
    assert math.isclose(vec.equity.iloc[-1], ev.equity.iloc[-1], rel_tol=1e-9)
    # Position arrays should be identical.
    assert (vec.positions == ev.positions).all()


# ---- no immediate re-entry after stop ----------------------------------

def test_no_reentry_same_bar_after_stop():
    # Stop fires on bar 2. Strategy still wants long, but we don't re-enter
    # the same bar — actual position is 0 on that bar.
    closes = [100, 100, 94, 100, 100]
    df = _ohlcv_from_close(closes)
    result = run_event_driven(
        long_and_hold, df, {},
        stop_config=StopConfig(stop_loss_pct=0.05),
        cost_config=ZERO_COSTS,
    )
    # Bar 2: stop fires; position closes that bar; actual_positions[2] == 0.
    assert result.positions.iloc[2] == 0.0
    # Bar 3+: re-entry can happen since target is 1.0 again.
    assert result.positions.iloc[3] == 1.0


# ---- StopConfig validation ---------------------------------------------

def test_stop_config_rejects_out_of_range_pct():
    with pytest.raises(ValueError):
        StopConfig(stop_loss_pct=-0.1)
    with pytest.raises(ValueError):
        StopConfig(trailing_stop_pct=1.5)
    # Zero is allowed (effectively disables).
    StopConfig(stop_loss_pct=0.0)


# ---- equity-curve invariant on stop-hit bar ----------------------------

def test_partial_bar_return_on_stop_hit_bar():
    # Entry on bar 0 at close=100. Bar 1: close=100 (held full bar, return 0).
    # Bar 2: open=99, low=94, close=92. SL=5% triggers at 95; open is 99 > 95,
    # so fill at the stop price = 95. Bar return = (95 - 100) / 100 = -5%.
    df = _ohlcv_from_close(
        [100, 100, 92],
        open_offset=[0, 0, -1],     # bar 2 open = 91 ... wait, 92-1=91
        high_offset=[0, 0, 0],
        low_offset=[0, 0, -6],      # bar 2 low = 86
    )
    # Adjust so bar 2 open > stop. Use a cleaner setup:
    n = 3
    close = np.array([100.0, 100.0, 92.0])
    open_ = np.array([100.0, 100.0, 99.0])   # bar 2 opens above stop
    high = np.array([100.0, 100.0, 99.0])
    low = np.array([100.0, 100.0, 86.0])     # low is well below
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": 1e6,
    }, index=idx)

    result = run_event_driven(
        long_and_hold, df, {},
        stop_config=StopConfig(stop_loss_pct=0.05),
        cost_config=ZERO_COSTS,
        starting_capital=100_000.0,
    )

    # Bar 2 partial return: stop fills at 95 (since open 99 > stop 95).
    # We held 1.0 * (95 / 100 - 1) = -5%.
    assert result.returns.iloc[2] == pytest.approx(-0.05)
    # Bar 0: no position held yet (target hasn't taken effect), return 0.
    assert result.returns.iloc[0] == pytest.approx(0.0)
    # Bar 1: held 1.0 for full bar, return = 0 (close was flat).
    assert result.returns.iloc[1] == pytest.approx(0.0)
    # Equity: 100k * (1 - 0.05) = 95k.
    assert result.equity.iloc[-1] == pytest.approx(95_000.0)
