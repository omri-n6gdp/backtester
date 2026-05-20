"""Phase 11.3 — tests for backend.deploy.runner.

Pure-compute tests. No T212, no DB.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.deploy.runner import (
    DUST_THRESHOLD_SHARES,
    OrderIntent,
    compute_intents,
    diff_to_intent,
    position_to_qty,
)
from tests.conftest import make_synthetic_ohlcv


# --- position_to_qty ------------------------------------------------------


def test_position_to_qty_long_half_position():
    # 0.5 signal × $5000 capital / $200 price = 12.5 shares
    assert position_to_qty(
        target_position=0.5, deployment_capital=5000.0, current_price=200.0,
    ) == 12.5


def test_position_to_qty_full_position():
    assert position_to_qty(
        target_position=1.0, deployment_capital=5000.0, current_price=200.0,
    ) == 25.0


def test_position_to_qty_zero_signal_returns_zero():
    assert position_to_qty(
        target_position=0.0, deployment_capital=5000.0, current_price=200.0,
    ) == 0.0


def test_position_to_qty_truncates_to_four_decimals():
    # 1/3 × 1000 / 7 = 47.619047... → rounded to 47.619
    qty = position_to_qty(
        target_position=1.0 / 3.0, deployment_capital=1000.0, current_price=7.0,
    )
    # Round to 4 decimals: 1000/21 = 47.6190476... → 47.619
    assert qty == round(1000.0 / 3.0 / 7.0, 4)


def test_position_to_qty_short_signal_returns_negative():
    """The runner doesn't clamp negative shares — the submitter is
    expected to refuse SELL-to-open if positions don't exist."""
    qty = position_to_qty(
        target_position=-1.0, deployment_capital=5000.0, current_price=200.0,
    )
    assert qty == -25.0


def test_position_to_qty_zero_price_raises():
    with pytest.raises(ValueError):
        position_to_qty(target_position=1.0, deployment_capital=1000.0, current_price=0.0)


# --- diff_to_intent -------------------------------------------------------


def test_diff_buy_when_target_above_current():
    intent = diff_to_intent(
        current_qty=0.0, target_qty=10.0, ticker="AAPL_US_EQ",
        order_type="market", limit_price=None, as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert intent is not None
    assert intent.side == "BUY"
    assert intent.qty == 10.0
    assert intent.ticker == "AAPL_US_EQ"
    assert intent.order_type == "market"


def test_diff_sell_when_target_below_current():
    intent = diff_to_intent(
        current_qty=10.0, target_qty=4.0, ticker="AAPL_US_EQ",
        order_type="limit", limit_price=200.0, as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert intent is not None
    assert intent.side == "SELL"
    assert intent.qty == 6.0
    assert intent.limit_price == 200.0


def test_diff_no_intent_when_matched():
    assert diff_to_intent(
        current_qty=10.0, target_qty=10.0, ticker="AAPL_US_EQ",
        order_type="market", limit_price=None, as_of_timestamp="x",
    ) is None


def test_diff_no_intent_below_dust():
    """Sub-dust deltas don't generate an order — T212 would 400 anyway."""
    assert diff_to_intent(
        current_qty=10.0, target_qty=10.0 + DUST_THRESHOLD_SHARES * 0.5,
        ticker="AAPL_US_EQ", order_type="market", limit_price=None, as_of_timestamp="x",
    ) is None


def test_diff_intent_dust_boundary_predicate():
    """Boundary: |delta| < threshold → skip; |delta| >= threshold → fire.
    Documents the strict-less-than predicate so a future refactor doesn't
    silently shift the cutoff."""
    just_under = diff_to_intent(
        current_qty=0.0, target_qty=DUST_THRESHOLD_SHARES * 0.5,
        ticker="AAPL_US_EQ", order_type="market", limit_price=None, as_of_timestamp="x",
    )
    assert just_under is None  # |delta| strictly below threshold → skipped

    at_threshold = diff_to_intent(
        current_qty=0.0, target_qty=DUST_THRESHOLD_SHARES,
        ticker="AAPL_US_EQ", order_type="market", limit_price=None, as_of_timestamp="x",
    )
    assert at_threshold is not None, "delta == threshold should fire (< is strict)"
    assert at_threshold.qty == DUST_THRESHOLD_SHARES

    just_over = diff_to_intent(
        current_qty=0.0, target_qty=DUST_THRESHOLD_SHARES * 2,
        ticker="AAPL_US_EQ", order_type="market", limit_price=None, as_of_timestamp="x",
    )
    assert just_over is not None
    assert just_over.qty == DUST_THRESHOLD_SHARES * 2


def test_diff_rationale_includes_target_and_current():
    intent = diff_to_intent(
        current_qty=2.5, target_qty=7.5, ticker="AAPL_US_EQ",
        order_type="market", limit_price=None, as_of_timestamp="x",
    )
    assert "7.5000" in intent.rationale
    assert "2.5000" in intent.rationale
    assert "BUY" in intent.rationale


# --- compute_intents end-to-end ------------------------------------------


# A no-state strategy that always wants 50% long. Importable name.
_LONG_HALF_STRATEGY = """import pandas as pd

def strategy(data, params):
    return pd.Series(0.5, index=data.index)
"""

# A strategy with a NaN warmup window then full long.
_WARMUP_STRATEGY = """import pandas as pd
import numpy as np

def strategy(data, params):
    out = pd.Series(np.nan, index=data.index)
    out.iloc[-1] = 1.0   # only the freshest bar has a real signal
    return out
"""

_NAN_AT_END_STRATEGY = """import pandas as pd
import numpy as np

def strategy(data, params):
    out = pd.Series(1.0, index=data.index)
    out.iloc[-1] = np.nan  # latest bar is NaN — runner treats as flat
    return out
"""


def _deployment(**overrides):
    base = {
        "id": "dep-test",
        "ticker": "AAPL_US_EQ",
        "params_json": json.dumps({}),
        "deployment_capital": 5000.0,
        "order_type": "market",
    }
    base.update(overrides)
    return base


def test_compute_intents_buy_when_flat_and_strategy_wants_long():
    data = make_synthetic_ohlcv(n=300)
    # Force a known last close so sizing is deterministic.
    data = data.copy()
    data.loc[data.index[-1], "close"] = 200.0

    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_LONG_HALF_STRATEGY,
        data=data,
        portfolio=[],
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert len(intents) == 1
    intent = intents[0]
    assert intent.side == "BUY"
    # target = 0.5 * 5000 / 200 = 12.5
    assert intent.qty == 12.5
    assert intent.order_type == "market"


def test_compute_intents_sell_when_held_more_than_target():
    data = make_synthetic_ohlcv(n=300)
    data = data.copy()
    data.loc[data.index[-1], "close"] = 200.0

    portfolio = [{"ticker": "AAPL_US_EQ", "quantity": 20.0}]
    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_LONG_HALF_STRATEGY,
        data=data,
        portfolio=portfolio,
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert len(intents) == 1
    # Target = 12.5 shares, current = 20 → SELL 7.5
    assert intents[0].side == "SELL"
    assert intents[0].qty == 7.5


def test_compute_intents_no_action_when_at_target():
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0

    portfolio = [{"ticker": "AAPL_US_EQ", "quantity": 12.5}]
    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_LONG_HALF_STRATEGY,
        data=data,
        portfolio=portfolio,
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert intents == []


def test_compute_intents_ignores_positions_for_other_tickers():
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0

    portfolio = [{"ticker": "MSFT_US_EQ", "quantity": 100.0}]  # different ticker
    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_LONG_HALF_STRATEGY,
        data=data,
        portfolio=portfolio,
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    # current_qty for AAPL_US_EQ is 0 (MSFT is unrelated); target is 12.5
    assert len(intents) == 1
    assert intents[0].side == "BUY"
    assert intents[0].qty == 12.5


def test_compute_intents_limit_order_uses_latest_close_as_limit():
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0

    intents = compute_intents(
        deployment=_deployment(order_type="limit"),
        strategy_code=_LONG_HALF_STRATEGY,
        data=data,
        portfolio=[],
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert intents[0].order_type == "limit"
    assert intents[0].limit_price == 200.0


def test_compute_intents_nan_latest_signal_treated_as_flat():
    """Strategy that returns NaN for the most recent bar is interpreted
    as 'go flat' (matches the backtest's .fillna(0.0) semantics).
    A flat target with no current position → no intent."""
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0

    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_NAN_AT_END_STRATEGY,
        data=data,
        portfolio=[],
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert intents == []


def test_compute_intents_nan_with_existing_holding_closes_position():
    """If the user holds shares in the deployment's ticker AND the
    strategy says NaN/flat, the deployment will SELL. This is the
    documented "deployment fully owns ticker" semantic."""
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0

    portfolio = [{"ticker": "AAPL_US_EQ", "quantity": 5.0}]
    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_NAN_AT_END_STRATEGY,
        data=data,
        portfolio=portfolio,
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    assert len(intents) == 1
    assert intents[0].side == "SELL"
    assert intents[0].qty == 5.0


def test_compute_intents_warmup_with_real_signal_at_end():
    """Strategy NaN through warmup then real signal on last bar →
    runner uses the real signal."""
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0

    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_WARMUP_STRATEGY,
        data=data,
        portfolio=[],
        as_of_timestamp="2026-05-19T21:00:00+00:00",
    )
    # Full long position: 1.0 * 5000 / 200 = 25 shares
    assert len(intents) == 1
    assert intents[0].side == "BUY"
    assert intents[0].qty == 25.0


def test_compute_intents_rejects_unknown_order_type():
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0
    with pytest.raises(ValueError):
        compute_intents(
            deployment=_deployment(order_type="stop"),  # not supported in Phase 11
            strategy_code=_LONG_HALF_STRATEGY,
            data=data,
            portfolio=[],
            as_of_timestamp="2026-05-19T21:00:00+00:00",
        )


def test_compute_intents_empty_data_raises():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError):
        compute_intents(
            deployment=_deployment(),
            strategy_code=_LONG_HALF_STRATEGY,
            data=empty,
            portfolio=[],
            as_of_timestamp="2026-05-19T21:00:00+00:00",
        )


# --- price_fx_factor (Phase 11.9 — currency-correct sizing) ---------------


def test_position_to_qty_applies_fx_factor():
    # capital is account-currency; price is instrument-currency. A 0.5
    # factor halves the effective price, so twice the shares.
    qty = position_to_qty(
        target_position=1.0, deployment_capital=2000.0,
        current_price=200.0, price_fx_factor=0.5,
    )
    assert qty == 20.0  # 2000 / (200 * 0.5)


def test_position_to_qty_factor_one_matches_no_factor():
    base = position_to_qty(
        target_position=1.0, deployment_capital=5000.0, current_price=200.0,
    )
    with_factor = position_to_qty(
        target_position=1.0, deployment_capital=5000.0,
        current_price=200.0, price_fx_factor=1.0,
    )
    assert base == with_factor == 25.0


def test_position_to_qty_rejects_non_positive_fx_factor():
    with pytest.raises(ValueError):
        position_to_qty(
            target_position=1.0, deployment_capital=1000.0,
            current_price=100.0, price_fx_factor=0.0,
        )


def test_compute_intents_applies_fx_factor():
    data = make_synthetic_ohlcv(n=300).copy()
    data.loc[data.index[-1], "close"] = 200.0
    # 0.5 signal * 5000 capital = 2500 target value; effective price
    # 200 * 0.5 = 100 -> 25 shares (vs 12.5 with no conversion).
    intents = compute_intents(
        deployment=_deployment(),
        strategy_code=_LONG_HALF_STRATEGY,
        data=data,
        portfolio=[],
        as_of_timestamp="2026-05-19T21:00:00+00:00",
        price_fx_factor=0.5,
    )
    assert len(intents) == 1
    assert intents[0].side == "BUY"
    assert intents[0].qty == 25.0
