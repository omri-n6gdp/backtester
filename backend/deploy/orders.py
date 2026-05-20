"""
Phase 11.4 — idempotent order submission.

Takes OrderIntent objects from `runner.py` and turns each into a row in
`orders_sent` plus (on success) a real T212 order. The idempotency
primitive is the UNIQUE constraint on `orders_sent.request_id_hash`:

  hash = sha256(deployment_id + as_of + ticker + side + order_type)[:16]

A duplicate insert raises `sqlite3.IntegrityError`, which the submitter
catches and treats as "already submitted; skip." The scheduler can
therefore fire `run_once()` arbitrarily many times for the same as-of
bar without double-submitting orders.

Failure semantics
-----------------
If the local DB insert succeeds but T212 rejects the order, the row is
marked `FAILED` with the error captured in `response_json`. A
subsequent `submit_intents()` call for the same intent will see the
hash collision and skip, treating FAILED as terminal for that intent.

This is intentional: a transient T212 failure won't be silently retried
by re-running the scheduler. The next bar's runner output has a
different `as_of_timestamp` → different hash → fresh submission. If the
strategy still wants the same position, it will be re-attempted via
fresh comparison against the current (still-empty) portfolio. So we
don't permanently miss trades; we just don't hot-loop a broken order.

Discrepancy after `T212TransportError`
--------------------------------------
If our client raised before getting T212's response (timeout, network),
we mark FAILED. T212 may or may not have actually accepted the order.
Reconciliation is implicit: the next bar's runner inspects the live
portfolio and adjusts. No special code path needed in Phase 11; Phase
12 (safeguards) tightens this.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.db import repo
from backend.deploy.runner import OrderIntent
from backend.t212 import client as t212_client

logger = logging.getLogger(__name__)


def compute_request_id_hash(*, deployment_id: str, intent: OrderIntent) -> str:
    """Stable 16-char hex idempotency hash for an order intent.

    The hash identifies "this deployment acted on this bar, this side":
    `deployment_id + as_of_timestamp + ticker + side + order_type`. It
    deliberately excludes `qty` and `limit_price` — both are derived from
    the live price, which drifts intraday, so including them would let a
    mid-day re-run of "Run now" slip a second order past the idempotency
    check (Phase 12.1 fix). Keyed this way, a deployment places exactly
    one order per (bar, ticker, side): a re-run within the same bar
    always collides and is skipped, and the next bar's fresh `as_of`
    goes through.
    """
    payload = "|".join([
        deployment_id,
        intent.as_of_timestamp,
        intent.ticker,
        intent.side,
        intent.order_type,
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SubmitResult:
    """Per-intent submission outcome. Returned in parallel to the input intents."""

    intent: OrderIntent
    order_id: str                # our internal UUID
    request_id_hash: str
    status: str                  # 'SUBMITTED' | 'SKIPPED_DUPLICATE' | 'FAILED'
    t212_order_id: int | None
    error: str | None            # populated only when status == 'FAILED'


def submit_intents(
    conn: sqlite3.Connection,
    *,
    deployment_id: str,
    intents: list[OrderIntent],
) -> list[SubmitResult]:
    """Submit each intent sequentially. Returns one SubmitResult per intent.

    The caller controls the transaction — this function does NOT commit.
    Use `with connect() as conn:` for the standard commit-on-success
    behavior. Each intent is processed independently; a failure on one
    does not affect subsequent intents in the batch.

    Args:
        conn: open SQLite connection.
        deployment_id: the parent deployment id (used in the idempotency hash).
        intents: list from `runner.compute_intents(...)`.
    """
    return [_submit_one(conn, deployment_id, intent) for intent in intents]


def _submit_one(
    conn: sqlite3.Connection, deployment_id: str, intent: OrderIntent,
) -> SubmitResult:
    request_id_hash = compute_request_id_hash(deployment_id=deployment_id, intent=intent)
    order_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    # Phase 1: try to claim the idempotency slot. Duplicate hash → skip.
    try:
        repo.insert_order(
            conn,
            order_id=order_id,
            deployment_id=deployment_id,
            request_id_hash=request_id_hash,
            as_of_timestamp=intent.as_of_timestamp,
            ticker=intent.ticker,
            side=intent.side,
            qty=intent.qty,
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            status="PENDING",
            created_at=now,
        )
    except sqlite3.IntegrityError:
        existing = repo.find_order_by_request_id(conn, request_id_hash)
        existing_id = existing["id"] if existing else order_id
        existing_t212_id = existing.get("t212_order_id") if existing else None
        logger.info(
            "Skipping duplicate intent (deployment=%s hash=%s existing=%s status=%s)",
            deployment_id, request_id_hash, existing_id,
            existing.get("status") if existing else "<unknown>",
        )
        return SubmitResult(
            intent=intent,
            order_id=existing_id,
            request_id_hash=request_id_hash,
            status="SKIPPED_DUPLICATE",
            t212_order_id=existing_t212_id,
            error=None,
        )

    # Phase 2a: validate the intent BEFORE calling T212. A malformed
    # intent (runner bug) becomes a FAILED row rather than a mid-batch
    # exception — that keeps the DB and T212 consistent when a batch
    # has multiple intents and one is bad.
    intent_error: str | None = None
    if intent.order_type not in ("market", "limit"):
        intent_error = f"unknown order_type {intent.order_type!r}"
    elif intent.order_type == "limit" and intent.limit_price is None:
        intent_error = "limit order requires limit_price"
    if intent_error is not None:
        error_payload = {"error_type": "ValueError", "message": intent_error}
        repo.update_order_status(
            conn, order_id, status="FAILED", response_json=json.dumps(error_payload),
        )
        logger.error(
            "Malformed intent for deployment %s (order %s): %s",
            deployment_id, order_id, intent_error,
        )
        return SubmitResult(
            intent=intent,
            order_id=order_id,
            request_id_hash=request_id_hash,
            status="FAILED",
            t212_order_id=None,
            error=intent_error,
        )

    # Phase 2b: place the order with T212. Sign convention: T212 infers
    # BUY/SELL from quantity sign (positive=buy, negative=sell).
    qty_signed = intent.qty if intent.side == "BUY" else -intent.qty
    try:
        if intent.order_type == "market":
            response = t212_client.place_market_order(intent.ticker, qty_signed)
        else:  # 'limit' — limit_price validated non-None above
            response = t212_client.place_limit_order(
                intent.ticker, qty_signed, intent.limit_price,
            )
    except t212_client.T212Error as exc:
        error_payload = {"error_type": type(exc).__name__, "message": str(exc)}
        repo.update_order_status(
            conn, order_id,
            status="FAILED",
            response_json=json.dumps(error_payload),
        )
        logger.warning(
            "T212 rejected order %s (deployment=%s): %s",
            order_id, deployment_id, exc,
        )
        return SubmitResult(
            intent=intent,
            order_id=order_id,
            request_id_hash=request_id_hash,
            status="FAILED",
            t212_order_id=None,
            error=str(exc),
        )

    # Phase 3: success — record T212's order id and full response.
    t212_order_id_raw = response.get("id") if isinstance(response, dict) else None
    t212_order_id = int(t212_order_id_raw) if t212_order_id_raw is not None else None
    repo.update_order_status(
        conn, order_id,
        status="SUBMITTED",
        t212_order_id=t212_order_id,
        response_json=json.dumps(response, default=str),
    )
    return SubmitResult(
        intent=intent,
        order_id=order_id,
        request_id_hash=request_id_hash,
        status="SUBMITTED",
        t212_order_id=t212_order_id,
        error=None,
    )
