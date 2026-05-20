"""
Phase 11.3 — deploy runner.

Pure-compute module. Given a deployment row, the live strategy code, the
current T212 portfolio, and a window of historical data ending at the
"as-of" bar, it produces the list of OrderIntent objects required to
move the live portfolio to the strategy's target position.

Crucially: this module **does not** persist, submit, or call T212. It
returns plain dataclasses. The 11.4 layer (`orders.py`) handles
idempotent persistence + submission.

Sizing model
------------
A strategy returns positions in [-1, +1] per bar. Live deployment
converts the most recent signal to an absolute share count via

    target_value = signal × deployment_capital
    target_qty   = round(target_value / latest_close, 4)

Rounding to 4 decimals matches T212's fractional-share precision for
most US equities. T212 will 400 with a clear `min-qty` message if the
instrument doesn't accept fractions; the submitter surfaces that.

NaN / warm-up handling
----------------------
Mirroring the backtest's `.shift(1).fillna(0.0)` semantics: a NaN
signal becomes 0.0 (flat). If the user manually holds shares in the
deployment's ticker while the strategy is in warm-up, the deployment
will SELL those shares to honor target=flat. Operationally, the
deployment fully owns positions in its ticker. Disarm if you intend to
hold shares manually.

Dust threshold
--------------
Position diffs below 1e-4 shares are dropped (no order). T212 rejects
sub-minimum quantities; dropping them locally avoids a guaranteed-fail
round-trip.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from backend.engine.validator import load_validated_strategy

logger = logging.getLogger(__name__)

# Below this share-count delta we consider current==target. T212's
# minimum trade quantity varies by instrument; 1e-4 is conservatively
# below all of them for fractional-share-enabled equities.
DUST_THRESHOLD_SHARES = 1e-4


@dataclass(frozen=True)
class OrderIntent:
    """A single order to submit. Sign-agnostic — `side` is explicit and
    `qty` is the unsigned magnitude."""

    ticker: str
    side: str               # 'BUY' | 'SELL'
    qty: float              # absolute shares; positive
    order_type: str         # 'market' | 'limit'
    limit_price: float | None
    as_of_timestamp: str    # ISO 8601 of the bar this decision was made on
    rationale: str          # human-readable provenance


# --- pure helpers ---------------------------------------------------------


def position_to_qty(
    *,
    target_position: float,
    deployment_capital: float,
    current_price: float,
    price_fx_factor: float = 1.0,
    max_position_value: float | None = None,
) -> float:
    """Convert a fractional position in [-1, +1] to an absolute share
    quantity. Truncates to 4 decimal places.

    Args:
        target_position: strategy signal in [-1, +1].
        deployment_capital: budget assigned to the deployment, in the
            T212 account's currency.
        current_price: latest close of the deployment's ticker, in the
            instrument's own currency (yfinance quotes a US stock in USD).
        price_fx_factor: multiplier that converts `current_price` from the
            instrument's currency into the account currency. 1.0 when the
            instrument already trades in the account currency; otherwise
            the FX rate (account-currency units per 1 instrument-currency
            unit). The scheduler computes it — see
            `scheduler.account_fx_factor`. Without it, a GBP account
            buying a USD stock is mis-sized by the FX rate.
        max_position_value: optional hard cap (account currency) on the
            position value — `target_value` is clamped to ±this before
            the share count is computed. None means no cap.

    Returns: signed share count (negative = short). The deployment layer
        does not currently support shorting equities on T212; the caller
        is expected to clamp to >= 0 before producing intents.
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be > 0, got {current_price}")
    if price_fx_factor <= 0:
        raise ValueError(f"price_fx_factor must be > 0, got {price_fx_factor}")
    # current_price is in the instrument's currency; convert it to the
    # account currency so it divides like-for-like into the capital.
    price_in_account_ccy = current_price * price_fx_factor
    target_value = target_position * deployment_capital
    # Safeguard (Phase 12.3): a hard cap on the position value, applied
    # after sizing — the deployment never builds a position worth more
    # than this, whatever the capital / signal / FX math produced.
    if max_position_value is not None:
        cap = abs(max_position_value)
        target_value = max(-cap, min(cap, target_value))
    return round(target_value / price_in_account_ccy, 4)


def diff_to_intent(
    *,
    current_qty: float,
    target_qty: float,
    ticker: str,
    order_type: str,
    limit_price: float | None,
    as_of_timestamp: str,
) -> OrderIntent | None:
    """Single-order delta from current to target. Returns None when the
    delta is below the dust threshold (no order needed).

    For ratchet-style deployments where we need to BUY 5 and SELL 3 of
    different tickers, callers loop over tickers and collect non-None
    results."""
    delta = target_qty - current_qty
    if abs(delta) < DUST_THRESHOLD_SHARES:
        return None
    side = "BUY" if delta > 0 else "SELL"
    return OrderIntent(
        ticker=ticker,
        side=side,
        qty=abs(delta),
        order_type=order_type,
        limit_price=limit_price,
        as_of_timestamp=as_of_timestamp,
        rationale=(
            f"target {target_qty:.4f} - current {current_qty:.4f} "
            f"= {delta:+.4f} ({side} {abs(delta):.4f})"
        ),
    )


# --- end-to-end driver ----------------------------------------------------


def _latest_signal(positions: pd.Series) -> float:
    """Return the most recent strategy position, mapping NaN → 0.0 (flat)
    to mirror the backtest's `.fillna(0.0)` semantics."""
    if positions is None or len(positions) == 0:
        raise ValueError("strategy returned empty positions series")
    last = positions.iloc[-1]
    if pd.isna(last):
        return 0.0
    return float(last)


def _current_qty_for_ticker(portfolio: list[dict[str, Any]], ticker: str) -> float:
    """T212 portfolio is a list of {ticker, quantity, ...} dicts. Match by
    instrument code (case-sensitive — T212's `ticker` field IS exact).
    Returns 0.0 if not held."""
    for pos in portfolio:
        if pos.get("ticker") == ticker:
            return float(pos.get("quantity", 0.0))
    return 0.0


def compute_intents(
    *,
    deployment: dict[str, Any],
    strategy_code: str,
    data: pd.DataFrame,
    portfolio: list[dict[str, Any]],
    as_of_timestamp: str,
    price_fx_factor: float = 1.0,
) -> list[OrderIntent]:
    """End-to-end runner: produce the order intents needed to move the
    live portfolio to the strategy's target as of the latest bar in
    `data`.

    Args:
        deployment: row from `repo.get_deployment(...)`.
        strategy_code: Python source of the strategy. Validated +
            executed in-process.
        data: OHLCV DataFrame ending at the as-of bar (no future bars).
            The runner reads `data['close'].iloc[-1]` for sizing.
        portfolio: current T212 positions (output of `t212_client.get_portfolio`).
        as_of_timestamp: ISO 8601 string identifying the bar (used as
            input to the idempotency hash; must be stable across retries).
        price_fx_factor: instrument-currency -> account-currency
            multiplier for sizing (see `scheduler.account_fx_factor`).

    Returns:
        List of OrderIntents. Empty when the strategy says "stay put" or
        when the diff is below the dust threshold.
    """
    if data is None or data.empty:
        raise ValueError("data is empty; cannot determine current price")

    fn: Callable[[pd.DataFrame, dict], pd.Series] = load_validated_strategy(strategy_code)
    params = json.loads(deployment["params_json"])

    positions = fn(data, params)
    target_position = _latest_signal(positions)

    current_price = float(data["close"].iloc[-1])
    max_pos = deployment.get("max_position_value")
    target_qty = position_to_qty(
        target_position=target_position,
        deployment_capital=float(deployment["deployment_capital"]),
        current_price=current_price,
        price_fx_factor=price_fx_factor,
        max_position_value=float(max_pos) if max_pos is not None else None,
    )
    current_qty = _current_qty_for_ticker(portfolio, deployment["ticker"])

    # Limit orders use the latest close as the trigger. Market orders
    # leave limit_price=None and accept T212's market fill.
    order_type = deployment["order_type"]
    if order_type not in ("market", "limit"):
        raise ValueError(f"unknown order_type {order_type!r}; expected 'market' or 'limit'")
    limit_price = current_price if order_type == "limit" else None

    intent = diff_to_intent(
        current_qty=current_qty,
        target_qty=target_qty,
        ticker=deployment["ticker"],
        order_type=order_type,
        limit_price=limit_price,
        as_of_timestamp=as_of_timestamp,
    )
    if intent is None:
        logger.info(
            "Deployment %s: no action (target=%.4f current=%.4f)",
            deployment["id"], target_qty, current_qty,
        )
        return []
    return [intent]
