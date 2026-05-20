"""Phase 11.4 — tests for backend.deploy.orders (idempotent submitter).

Uses the same per-test DB fixture as test_db_deployments. T212 calls are
monkeypatched — no live network.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.db import repo
from backend.db.conn import connect
from backend.deploy.orders import (
    SubmitResult,
    compute_request_id_hash,
    reconcile_orders,
    submit_intents,
)
from backend.deploy.runner import OrderIntent
from backend.t212 import client as t212_client


# --- fixture (mirrors test_db_deployments.fresh_db) -----------------------


@pytest.fixture
def fresh_db_with_deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Per-test SQLite + a seeded strategy + a deployment row."""
    data_root = tmp_path / "btdata"
    db_dir = data_root / "results"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "results.db"

    monkeypatch.setattr("backend.storage.DATA_ROOT", data_root, raising=False)
    monkeypatch.setattr("backend.storage.DB_DIR", db_dir, raising=False)
    monkeypatch.setattr("backend.storage.DB_PATH", db_path, raising=False)

    from backend.db.conn import migrate
    migrate()

    with connect() as conn:
        repo.insert_strategy(
            conn, strategy_id="strat-seed", name="Test", description="",
            code_path="/tmp/x.py", created_at=_now(),
        )
        repo.insert_deployment(
            conn, deployment_id="dep-1", strategy_id="strat-seed",
            ticker="AAPL_US_EQ", timeframe="1d",
            params_json=json.dumps({}), deployment_capital=5000.0,
            cost_config_json=json.dumps({}), order_type="market",
            schedule_cron="5 21 * * 1-5", armed=False, created_at=_now(),
        )
    return "dep-1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_intent(**overrides) -> OrderIntent:
    base = {
        "ticker": "AAPL_US_EQ",
        "side": "BUY",
        "qty": 1.0,
        "order_type": "market",
        "limit_price": None,
        "as_of_timestamp": "2026-05-19T21:00:00+00:00",
        "rationale": "test",
    }
    base.update(overrides)
    return OrderIntent(**base)


_T212_RESPONSE = {
    "id": 49150662798,
    "type": "MARKET",
    "ticker": "AAPL_US_EQ",
    "quantity": 1.0,
    "filledQuantity": 0,
    "status": "NEW",
    "side": "BUY",
    "createdAt": "2026-05-19T21:00:01+00:00",
}


# --- compute_request_id_hash ----------------------------------------------


def test_hash_is_deterministic():
    intent = _make_intent()
    h1 = compute_request_id_hash(deployment_id="dep-1", intent=intent)
    h2 = compute_request_id_hash(deployment_id="dep-1", intent=intent)
    assert h1 == h2
    assert len(h1) == 16


def test_hash_differs_when_deployment_id_changes():
    intent = _make_intent()
    a = compute_request_id_hash(deployment_id="dep-A", intent=intent)
    b = compute_request_id_hash(deployment_id="dep-B", intent=intent)
    assert a != b


def test_hash_stable_when_qty_changes():
    """Phase 12.1: qty drifts with the live price, so it must NOT affect
    the idempotency hash — otherwise a mid-day re-run double-orders."""
    a = compute_request_id_hash(deployment_id="dep-1", intent=_make_intent(qty=1.0))
    b = compute_request_id_hash(deployment_id="dep-1", intent=_make_intent(qty=2.0))
    assert a == b


def test_hash_differs_when_side_flips():
    a = compute_request_id_hash(deployment_id="dep-1", intent=_make_intent(side="BUY"))
    b = compute_request_id_hash(deployment_id="dep-1", intent=_make_intent(side="SELL"))
    assert a != b


def test_hash_differs_when_as_of_changes():
    a = compute_request_id_hash(
        deployment_id="dep-1",
        intent=_make_intent(as_of_timestamp="2026-05-19T21:00:00+00:00"),
    )
    b = compute_request_id_hash(
        deployment_id="dep-1",
        intent=_make_intent(as_of_timestamp="2026-05-20T21:00:00+00:00"),
    )
    assert a != b


def test_hash_stable_when_limit_price_changes():
    """Phase 12.1: limit_price (= the live close) is excluded from the
    hash for the same reason as qty."""
    a = compute_request_id_hash(
        deployment_id="dep-1",
        intent=_make_intent(order_type="limit", limit_price=200.0),
    )
    b = compute_request_id_hash(
        deployment_id="dep-1",
        intent=_make_intent(order_type="limit", limit_price=201.0),
    )
    assert a == b


# --- submit_intents happy paths -------------------------------------------


def test_submit_one_intent_happy_path(fresh_db_with_deployment, monkeypatch):
    captured = []

    def fake_place_market(ticker, quantity):
        captured.append((ticker, quantity))
        return _T212_RESPONSE

    monkeypatch.setattr(t212_client, "place_market_order", fake_place_market)

    with connect() as conn:
        results = submit_intents(
            conn, deployment_id=fresh_db_with_deployment, intents=[_make_intent()],
        )

    assert len(results) == 1
    r = results[0]
    assert r.status == "SUBMITTED"
    assert r.t212_order_id == 49150662798
    assert r.error is None
    # T212 received the BUY as positive quantity.
    assert captured == [("AAPL_US_EQ", 1.0)]

    # Row is persisted with the right state.
    with connect() as conn:
        row = repo.find_order_by_request_id(conn, r.request_id_hash)
    assert row["status"] == "SUBMITTED"
    assert row["t212_order_id"] == 49150662798
    assert json.loads(row["response_json"])["status"] == "NEW"


def test_submit_sell_intent_passes_negative_quantity(fresh_db_with_deployment, monkeypatch):
    captured = []

    def fake_place_market(ticker, quantity):
        captured.append((ticker, quantity))
        return {**_T212_RESPONSE, "side": "SELL", "quantity": -2.5}

    monkeypatch.setattr(t212_client, "place_market_order", fake_place_market)

    with connect() as conn:
        submit_intents(
            conn, deployment_id=fresh_db_with_deployment,
            intents=[_make_intent(side="SELL", qty=2.5)],
        )
    assert captured == [("AAPL_US_EQ", -2.5)]


def test_submit_limit_intent_calls_place_limit_order(fresh_db_with_deployment, monkeypatch):
    captured = []

    def fake_place_limit(ticker, quantity, limit_price, time_validity="DAY"):
        captured.append((ticker, quantity, limit_price, time_validity))
        return {**_T212_RESPONSE, "type": "LIMIT", "limitPrice": 200.0}

    monkeypatch.setattr(t212_client, "place_limit_order", fake_place_limit)

    with connect() as conn:
        submit_intents(
            conn, deployment_id=fresh_db_with_deployment,
            intents=[_make_intent(order_type="limit", limit_price=200.0)],
        )
    assert captured == [("AAPL_US_EQ", 1.0, 200.0, "DAY")]


def test_submit_limit_without_price_marks_failed(fresh_db_with_deployment, monkeypatch):
    """The runner always sets limit_price for limit orders; the submitter
    defends against a missing one by recording a FAILED row instead of
    raising mid-batch (which could roll back a sibling intent's real
    T212 order)."""
    intent = _make_intent(order_type="limit", limit_price=None)
    with connect() as conn:
        results = submit_intents(
            conn, deployment_id=fresh_db_with_deployment, intents=[intent],
        )
    assert results[0].status == "FAILED"
    assert "limit_price" in (results[0].error or "")
    with connect() as conn:
        row = repo.find_order_by_request_id(conn, results[0].request_id_hash)
    assert row["status"] == "FAILED"
    assert row["t212_order_id"] is None


def test_bad_intent_does_not_roll_back_sibling_order(fresh_db_with_deployment, monkeypatch):
    """A malformed intent in a multi-intent batch must NOT undo a
    sibling intent's already-placed T212 order. The good one stays
    SUBMITTED; the bad one is FAILED."""
    placed = []

    def fake_place_market(ticker, quantity):
        placed.append((ticker, quantity))
        return _T212_RESPONSE

    monkeypatch.setattr(t212_client, "place_market_order", fake_place_market)

    good = _make_intent(qty=1.0)
    bad = _make_intent(qty=2.0, order_type="limit", limit_price=None)
    with connect() as conn:
        results = submit_intents(
            conn, deployment_id=fresh_db_with_deployment, intents=[good, bad],
        )
    assert results[0].status == "SUBMITTED"
    assert results[1].status == "FAILED"
    assert len(placed) == 1, "only the good intent should reach T212"

    with connect() as conn:
        good_row = repo.find_order_by_request_id(conn, results[0].request_id_hash)
        bad_row = repo.find_order_by_request_id(conn, results[1].request_id_hash)
    assert good_row["status"] == "SUBMITTED"
    assert bad_row["status"] == "FAILED"


def test_submit_multiple_intents_all_recorded(fresh_db_with_deployment, monkeypatch):
    calls = []

    def fake_place_market(ticker, quantity):
        calls.append((ticker, quantity))
        return {**_T212_RESPONSE, "id": 100 + len(calls)}

    monkeypatch.setattr(t212_client, "place_market_order", fake_place_market)

    # A realistic multi-intent batch differs by ticker (one diff per
    # ticker), so each intent has a distinct idempotency hash.
    intents = [
        _make_intent(ticker="AAPL_US_EQ", qty=1.0),
        _make_intent(ticker="MSFT_US_EQ", qty=2.0),
        _make_intent(ticker="GOOGL_US_EQ", qty=3.0),
    ]
    with connect() as conn:
        results = submit_intents(
            conn, deployment_id=fresh_db_with_deployment, intents=intents,
        )
    assert [r.status for r in results] == ["SUBMITTED"] * 3
    assert [r.t212_order_id for r in results] == [101, 102, 103]
    assert calls == [("AAPL_US_EQ", 1.0), ("MSFT_US_EQ", 2.0), ("GOOGL_US_EQ", 3.0)]


# --- idempotency ----------------------------------------------------------


def test_duplicate_submission_is_skipped(fresh_db_with_deployment, monkeypatch):
    """Same intent submitted twice → second call returns SKIPPED_DUPLICATE
    and T212 is called only once."""
    calls = 0

    def fake_place_market(ticker, quantity):
        nonlocal calls
        calls += 1
        return _T212_RESPONSE

    monkeypatch.setattr(t212_client, "place_market_order", fake_place_market)

    intent = _make_intent()
    with connect() as conn:
        first = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[intent])
    with connect() as conn:
        second = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[intent])

    assert first[0].status == "SUBMITTED"
    assert second[0].status == "SKIPPED_DUPLICATE"
    assert second[0].order_id == first[0].order_id, \
        "duplicate should report the existing order's id"
    assert second[0].t212_order_id == first[0].t212_order_id
    assert calls == 1, "T212 should be called exactly once across the two submissions"


def test_failed_submission_is_terminal_for_that_hash(fresh_db_with_deployment, monkeypatch):
    """A T212 failure marks the row FAILED. A re-submit with the same
    intent hash returns SKIPPED_DUPLICATE (no retry) — by design.
    The next bar's intent has a fresh hash and will be tried anew."""
    def raises(ticker, quantity):
        raise t212_client.T212UpstreamError("T212 is down")

    monkeypatch.setattr(t212_client, "place_market_order", raises)

    intent = _make_intent()
    with connect() as conn:
        first = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[intent])
    assert first[0].status == "FAILED"
    assert "T212 is down" in (first[0].error or "")

    # Same intent submitted again -- skipped via the FAILED row's hash collision.
    with connect() as conn:
        second = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[intent])
    assert second[0].status == "SKIPPED_DUPLICATE"

    # The persisted row has status FAILED and the error captured in response_json.
    with connect() as conn:
        row = repo.find_order_by_request_id(conn, first[0].request_id_hash)
    assert row["status"] == "FAILED"
    assert row["t212_order_id"] is None
    error_blob = json.loads(row["response_json"])
    assert error_blob["error_type"] == "T212UpstreamError"
    assert "T212 is down" in error_blob["message"]


def test_failure_then_fresh_intent_with_new_as_of_proceeds(fresh_db_with_deployment, monkeypatch):
    """Tomorrow's run has a different as_of_timestamp → fresh hash →
    submitted independently of yesterday's FAILED row."""
    call_log = []

    def stub(ticker, quantity):
        call_log.append(("market", ticker, quantity))
        if len(call_log) == 1:
            raise t212_client.T212UpstreamError("rate limited")
        return _T212_RESPONSE

    monkeypatch.setattr(t212_client, "place_market_order", stub)

    intent_today = _make_intent(as_of_timestamp="2026-05-19T21:00:00+00:00")
    intent_tomorrow = _make_intent(as_of_timestamp="2026-05-20T21:00:00+00:00")

    with connect() as conn:
        r1 = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[intent_today])
    with connect() as conn:
        r2 = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[intent_tomorrow])

    assert r1[0].status == "FAILED"
    assert r2[0].status == "SUBMITTED"
    assert r2[0].t212_order_id == 49150662798


# --- error variants -------------------------------------------------------


def test_auth_error_recorded_with_error_type(fresh_db_with_deployment, monkeypatch):
    def raises(ticker, quantity):
        raise t212_client.T212AuthError("bad credentials")

    monkeypatch.setattr(t212_client, "place_market_order", raises)

    intent = _make_intent()
    with connect() as conn:
        results = submit_intents(
            conn, deployment_id=fresh_db_with_deployment, intents=[intent],
        )
    assert results[0].status == "FAILED"
    with connect() as conn:
        row = repo.find_order_by_request_id(conn, results[0].request_id_hash)
    error_blob = json.loads(row["response_json"])
    assert error_blob["error_type"] == "T212AuthError"


def test_rate_limited_recorded_as_failed(fresh_db_with_deployment, monkeypatch):
    def raises(ticker, quantity):
        raise t212_client.T212RateLimited("slow down", retry_after=5.0)

    monkeypatch.setattr(t212_client, "place_market_order", raises)

    with connect() as conn:
        results = submit_intents(
            conn, deployment_id=fresh_db_with_deployment, intents=[_make_intent()],
        )
    assert results[0].status == "FAILED"
    assert "slow down" in (results[0].error or "")


def test_empty_intent_list_returns_empty_results(fresh_db_with_deployment):
    with connect() as conn:
        results = submit_intents(
            conn, deployment_id=fresh_db_with_deployment, intents=[],
        )
    assert results == []


def test_submit_skips_a_requantified_intent_for_the_same_bar(
    fresh_db_with_deployment, monkeypatch,
):
    """Phase 12.1: a mid-day re-run recomputes a slightly different qty
    off the drifting live price; that must NOT slip a second order — the
    same (bar, ticker, side) is one order."""
    calls = 0

    def fake_place(ticker, quantity):
        nonlocal calls
        calls += 1
        return _T212_RESPONSE

    monkeypatch.setattr(t212_client, "place_market_order", fake_place)

    first = _make_intent(qty=6.6, as_of_timestamp="2026-05-20T00:00:00+00:00")
    requantified = _make_intent(qty=6.5, as_of_timestamp="2026-05-20T00:00:00+00:00")

    with connect() as conn:
        r1 = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[first])
    with connect() as conn:
        r2 = submit_intents(conn, deployment_id=fresh_db_with_deployment, intents=[requantified])

    assert r1[0].status == "SUBMITTED"
    assert r2[0].status == "SKIPPED_DUPLICATE"
    assert calls == 1, "the re-quantified same-bar intent must not place a 2nd order"


# --- reconcile_orders (Phase 12.6) ---------------------------------------


def _submit_one_order(deployment_id, t212_id, monkeypatch):
    """Place one SUBMITTED order and return its internal order id."""
    monkeypatch.setattr(t212_client, "place_market_order",
                        lambda tk, q: {**_T212_RESPONSE, "id": t212_id})
    with connect() as conn:
        results = submit_intents(conn, deployment_id=deployment_id, intents=[_make_intent()])
    return results[0].order_id


def test_reconcile_marks_a_filled_order(fresh_db_with_deployment, monkeypatch):
    order_id = _submit_one_order(fresh_db_with_deployment, 555, monkeypatch)
    monkeypatch.setattr(t212_client, "get_order",
                        lambda oid: {"id": oid, "status": "FILLED",
                                     "filledQuantity": 1.0, "fillPrice": 201.5})
    with connect() as conn:
        changed = reconcile_orders(conn, deployment_id=fresh_db_with_deployment)
    assert changed == 1
    with connect() as conn:
        orders = repo.list_orders_for_deployment(conn, fresh_db_with_deployment)
        fills = repo.list_fills_for_order(conn, order_id)
    assert orders[0]["status"] == "FILLED"
    assert len(fills) == 1
    assert fills[0]["filled_qty"] == 1.0
    assert fills[0]["filled_price"] == 201.5


def test_reconcile_marks_a_cancelled_order(fresh_db_with_deployment, monkeypatch):
    _submit_one_order(fresh_db_with_deployment, 556, monkeypatch)
    monkeypatch.setattr(t212_client, "get_order",
                        lambda oid: {"id": oid, "status": "CANCELLED"})
    with connect() as conn:
        reconcile_orders(conn, deployment_id=fresh_db_with_deployment)
        orders = repo.list_orders_for_deployment(conn, fresh_db_with_deployment)
    assert orders[0]["status"] == "CANCELLED"


def test_reconcile_leaves_a_working_order_submitted(fresh_db_with_deployment, monkeypatch):
    _submit_one_order(fresh_db_with_deployment, 557, monkeypatch)
    monkeypatch.setattr(t212_client, "get_order", lambda oid: {"id": oid, "status": "NEW"})
    with connect() as conn:
        changed = reconcile_orders(conn, deployment_id=fresh_db_with_deployment)
        orders = repo.list_orders_for_deployment(conn, fresh_db_with_deployment)
    assert changed == 0
    assert orders[0]["status"] == "SUBMITTED"


def test_reconcile_skips_an_order_when_t212_errors(fresh_db_with_deployment, monkeypatch):
    _submit_one_order(fresh_db_with_deployment, 558, monkeypatch)

    def boom(oid):
        raise t212_client.T212UpstreamError("T212 down")

    monkeypatch.setattr(t212_client, "get_order", boom)
    with connect() as conn:
        changed = reconcile_orders(conn, deployment_id=fresh_db_with_deployment)
        orders = repo.list_orders_for_deployment(conn, fresh_db_with_deployment)
    assert changed == 0
    assert orders[0]["status"] == "SUBMITTED"  # unchanged; the error was swallowed
