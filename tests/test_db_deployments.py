"""Phase 11.2 — repo-layer tests for deployments / orders_sent / fills_received.

Tests bypass the FastAPI layer and operate directly on backend.db.repo
against a fresh per-test SQLite database. Pattern mirrors tests/conftest.py's
`api_client` fixture, minus the FastAPI app.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.db import repo
from backend.db.conn import connect


@pytest.fixture
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Per-test SQLite. Yields nothing — tests use `with connect()` directly."""
    data_root = tmp_path / "btdata"
    db_dir = data_root / "results"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "results.db"

    monkeypatch.setattr("backend.storage.DATA_ROOT", data_root, raising=False)
    monkeypatch.setattr("backend.storage.DB_DIR", db_dir, raising=False)
    monkeypatch.setattr("backend.storage.DB_PATH", db_path, raising=False)

    from backend.db.conn import migrate
    migrate()

    # Seed a parent strategy so FK constraints don't bite the deployment
    # inserts. (Strategies have a real FK relationship into deployments.)
    with connect() as conn:
        repo.insert_strategy(
            conn,
            strategy_id="strat-seed",
            name="Test strategy",
            description="",
            code_path="/tmp/seed.py",
            created_at="2026-05-19T00:00:00+00:00",
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deployment_kwargs(**overrides):
    base = {
        "deployment_id": "dep-1",
        "strategy_id": "strat-seed",
        "ticker": "AAPL_US_EQ",
        "timeframe": "1d",
        "params_json": json.dumps({"short": 50, "long": 200}),
        "deployment_capital": 5000.0,
        "cost_config_json": json.dumps({"slippage_bps": 5.0}),
        "order_type": "market",
        "schedule_cron": "5 21 * * 1-5",
        "armed": False,
        "created_at": _now(),
    }
    base.update(overrides)
    return base


# --- deployments ----------------------------------------------------------


def test_insert_and_get_deployment_round_trip(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
    with connect() as conn:
        row = repo.get_deployment(conn, "dep-1")
    assert row is not None
    assert row["id"] == "dep-1"
    assert row["ticker"] == "AAPL_US_EQ"
    assert row["deployment_capital"] == 5000.0
    assert row["armed"] == 0  # stored as 0/1, not bool
    assert row["last_run_at"] is None


def test_get_deployment_returns_none_for_unknown(fresh_db):
    with connect() as conn:
        assert repo.get_deployment(conn, "nope") is None


def test_list_deployments_filters_by_strategy(fresh_db):
    with connect() as conn:
        repo.insert_strategy(
            conn, strategy_id="strat-other", name="Other", description="",
            code_path="/tmp/other.py", created_at=_now(),
        )
        repo.insert_deployment(conn, **_deployment_kwargs(deployment_id="dep-1"))
        repo.insert_deployment(conn, **_deployment_kwargs(
            deployment_id="dep-2", strategy_id="strat-other"))
    with connect() as conn:
        seed_only = repo.list_deployments(conn, strategy_id="strat-seed")
        other_only = repo.list_deployments(conn, strategy_id="strat-other")
    assert [d["id"] for d in seed_only] == ["dep-1"]
    assert [d["id"] for d in other_only] == ["dep-2"]


def test_list_deployments_armed_only_filter(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs(deployment_id="dep-disarmed", armed=False))
        repo.insert_deployment(conn, **_deployment_kwargs(deployment_id="dep-armed", armed=True))
    with connect() as conn:
        all_deps = repo.list_deployments(conn)
        armed = repo.list_deployments(conn, armed_only=True)
    assert {d["id"] for d in all_deps} == {"dep-armed", "dep-disarmed"}
    assert [d["id"] for d in armed] == ["dep-armed"]


def test_set_deployment_armed_toggles_flag(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs(armed=False))
    with connect() as conn:
        assert repo.set_deployment_armed(conn, "dep-1", True) is True
    with connect() as conn:
        assert repo.get_deployment(conn, "dep-1")["armed"] == 1
        repo.set_deployment_armed(conn, "dep-1", False)
    with connect() as conn:
        assert repo.get_deployment(conn, "dep-1")["armed"] == 0


def test_set_deployment_armed_returns_false_when_missing(fresh_db):
    with connect() as conn:
        assert repo.set_deployment_armed(conn, "no-such-id", True) is False


def test_update_deployment_last_run(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
    with connect() as conn:
        repo.update_deployment_last_run(conn, "dep-1", "2026-05-19T21:05:00+00:00")
    with connect() as conn:
        assert repo.get_deployment(conn, "dep-1")["last_run_at"] == "2026-05-19T21:05:00+00:00"


# --- orders ---------------------------------------------------------------


def _order_kwargs(**overrides):
    base = {
        "order_id": "ord-1",
        "deployment_id": "dep-1",
        "request_id_hash": "abc1234567890abc",
        "as_of_timestamp": "2026-05-19T21:00:00+00:00",
        "ticker": "AAPL_US_EQ",
        "side": "BUY",
        "qty": 1.5,
        "order_type": "market",
        "limit_price": None,
        "status": "PENDING",
        "created_at": _now(),
    }
    base.update(overrides)
    return base


def test_insert_and_find_order_by_request_id(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
        repo.insert_order(conn, **_order_kwargs())
    with connect() as conn:
        hit = repo.find_order_by_request_id(conn, "abc1234567890abc")
        miss = repo.find_order_by_request_id(conn, "deadbeef")
    assert hit is not None and hit["id"] == "ord-1"
    assert hit["qty"] == 1.5
    assert hit["side"] == "BUY"
    assert hit["limit_price"] is None
    assert miss is None


def test_duplicate_request_id_hash_raises(fresh_db):
    """The idempotency primitive: inserting the same request_id_hash twice
    raises IntegrityError. The deploy layer relies on this."""
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
        repo.insert_order(conn, **_order_kwargs(order_id="ord-1"))
    with connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_order(conn, **_order_kwargs(order_id="ord-2"))  # same hash


def test_update_order_status_records_t212_id_and_response(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
        repo.insert_order(conn, **_order_kwargs())
    with connect() as conn:
        ok = repo.update_order_status(
            conn, "ord-1", status="SUBMITTED",
            t212_order_id=49150662798,
            response_json='{"id": 49150662798, "status": "NEW"}',
        )
    assert ok is True
    with connect() as conn:
        row = repo.find_order_by_request_id(conn, "abc1234567890abc")
    assert row["status"] == "SUBMITTED"
    assert row["t212_order_id"] == 49150662798
    assert json.loads(row["response_json"])["status"] == "NEW"


def test_update_order_status_supports_status_only(fresh_db):
    """Status-only update doesn't blow away t212_order_id from a prior call."""
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
        repo.insert_order(conn, **_order_kwargs())
        repo.update_order_status(conn, "ord-1", status="SUBMITTED", t212_order_id=123)
    with connect() as conn:
        repo.update_order_status(conn, "ord-1", status="FILLED")
    with connect() as conn:
        row = repo.find_order_by_request_id(conn, "abc1234567890abc")
    assert row["status"] == "FILLED"
    assert row["t212_order_id"] == 123, "prior t212_order_id should survive"


def test_list_orders_for_deployment_orders_newest_first(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
        repo.insert_order(conn, **_order_kwargs(
            order_id="ord-old", request_id_hash="h-old", created_at="2026-05-18T00:00:00+00:00"))
        repo.insert_order(conn, **_order_kwargs(
            order_id="ord-new", request_id_hash="h-new", created_at="2026-05-19T00:00:00+00:00"))
    with connect() as conn:
        orders = repo.list_orders_for_deployment(conn, "dep-1")
    assert [o["id"] for o in orders] == ["ord-new", "ord-old"]


# --- fills ----------------------------------------------------------------


def test_insert_and_list_fills_for_order(fresh_db):
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
        repo.insert_order(conn, **_order_kwargs())
        fid = repo.insert_fill(
            conn,
            order_id="ord-1", t212_order_id=42,
            filled_qty=1.5, filled_price=200.05,
            filled_at="2026-05-19T21:05:01+00:00",
            raw_json='{"id": 42, "filledQuantity": 1.5}',
        )
    assert fid > 0
    with connect() as conn:
        fills = repo.list_fills_for_order(conn, "ord-1")
    assert len(fills) == 1
    assert fills[0]["filled_qty"] == 1.5
    assert fills[0]["filled_price"] == 200.05


def test_deleting_deployment_cascades_to_orders_and_fills(fresh_db):
    """ON DELETE CASCADE keeps the schema honest. Deleting a deployment
    removes its orders + their fills, no manual cleanup required."""
    with connect() as conn:
        repo.insert_deployment(conn, **_deployment_kwargs())
        repo.insert_order(conn, **_order_kwargs())
        repo.insert_fill(
            conn, order_id="ord-1", t212_order_id=42, filled_qty=1.5,
            filled_price=200.0, filled_at="2026-05-19T21:05:01+00:00",
            raw_json="{}",
        )
        conn.execute("DELETE FROM deployments WHERE id = ?", ("dep-1",))
    with connect() as conn:
        assert repo.get_deployment(conn, "dep-1") is None
        assert repo.find_order_by_request_id(conn, "abc1234567890abc") is None
        assert repo.list_fills_for_order(conn, "ord-1") == []
