"""Phase 11.5 — tests for backend.deploy.scheduler.

Two halves:
  * `run_deployment` — the job logic. Exercised end-to-end against a real
    DB and the real runner + submitter, with only the network seams
    (prices, T212 portfolio, T212 order placement, instrument lookup)
    monkeypatched. No live network, no real scheduler thread.
  * registration helpers — exercised against a non-started scheduler,
    which APScheduler fully supports (jobs sit "pending").
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from apscheduler.triggers.cron import CronTrigger

from backend.db import repo
from backend.db.conn import connect
from backend.deploy import scheduler
from backend.t212 import client as t212_client
from backend.t212 import instruments
from tests.conftest import make_synthetic_ohlcv


# --- strategy fixtures ----------------------------------------------------

# Always wants 50% long. Deterministic, no warm-up.
_LONG_HALF_STRATEGY = """import pandas as pd

def strategy(data, params):
    return pd.Series(0.5, index=data.index)
"""

# Always flat.
_FLAT_STRATEGY = """import pandas as pd

def strategy(data, params):
    return pd.Series(0.0, index=data.index)
"""

# A plausible T212 order response (mirrors test_deploy_orders).
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


# --- fixtures & helpers ---------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_scheduler():
    """Every test gets a clean scheduler singleton, shut down afterwards."""
    scheduler._reset_scheduler()
    yield
    scheduler._reset_scheduler()


@pytest.fixture
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test SQLite, migrated. Returns tmp_path for strategy-file writes."""
    data_root = tmp_path / "btdata"
    db_dir = data_root / "results"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "results.db"

    monkeypatch.setattr("backend.storage.DATA_ROOT", data_root, raising=False)
    monkeypatch.setattr("backend.storage.DB_DIR", db_dir, raising=False)
    monkeypatch.setattr("backend.storage.DB_PATH", db_path, raising=False)

    from backend.db.conn import migrate
    migrate()
    return tmp_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_deployment(
    tmp_path: Path,
    *,
    deployment_id: str = "dep-1",
    armed: bool = True,
    strategy_code: str = _LONG_HALF_STRATEGY,
    ticker: str = "AAPL_US_EQ",
    timeframe: str = "1d",
    capital: float = 5000.0,
    order_type: str = "market",
    cron: str = "5 21 * * 1-5",
    params: dict | None = None,
    max_loss_value: float | None = None,
) -> str:
    """Insert a strategy (+ its code file) and a deployment row. Returns
    the deployment id."""
    strategy_id = f"strat-{deployment_id}"
    code_file = tmp_path / f"{strategy_id}.py"
    code_file.write_text(strategy_code, encoding="utf-8")
    with connect() as conn:
        repo.insert_strategy(
            conn, strategy_id=strategy_id, name="Test Strategy",
            description="", code_path=str(code_file), created_at=_now(),
        )
        repo.insert_deployment(
            conn, deployment_id=deployment_id, strategy_id=strategy_id,
            ticker=ticker, timeframe=timeframe,
            params_json=json.dumps(params or {}), deployment_capital=capital,
            cost_config_json=json.dumps({}), order_type=order_type,
            schedule_cron=cron, armed=armed, created_at=_now(),
            max_loss_value=max_loss_value,
        )
    return deployment_id


def _recording_prices(df: pd.DataFrame):
    """A fake get_prices that records each call and returns `df`."""
    def fake(ticker, timeframe, start, end):
        fake.calls.append((ticker, timeframe, start, end))
        return df
    fake.calls = []  # type: ignore[attr-defined]
    return fake


# --- run_deployment: happy path ------------------------------------------


def test_run_deployment_happy_path_submits_order(fresh_db, monkeypatch):
    dep_id = _seed_deployment(fresh_db, armed=True)

    df = make_synthetic_ohlcv(n=300).copy()
    df.loc[df.index[-1], "close"] = 200.0
    fake_prices = _recording_prices(df)
    monkeypatch.setattr(scheduler, "get_prices", fake_prices)
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(scheduler, "account_fx_factor", lambda t: 1.0)
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])
    # Phase 14 — the precheck calls T212; keep it off the network.
    monkeypatch.setattr(t212_client, "get_account_cash", lambda: {"free": 1_000_000.0})

    placed = []

    def fake_place(ticker, quantity):
        placed.append((ticker, quantity))
        return {**_T212_RESPONSE, "ticker": ticker, "quantity": quantity}

    monkeypatch.setattr(t212_client, "place_market_order", fake_place)

    result = scheduler.run_deployment(dep_id)

    assert result.status == "OK"
    assert result.intent_count == 1
    assert len(result.submit_results) == 1
    assert result.submit_results[0].status == "SUBMITTED"
    # 0.5 signal * $5000 / $200 = 12.5 shares BUY, placed under the T212 code.
    assert placed == [("AAPL_US_EQ", 12.5)]
    # Prices were fetched for the resolved yfinance symbol, not the T212 code.
    assert fake_prices.calls[0][0] == "AAPL"
    # as_of is the last bar's timestamp, not wall-clock now.
    assert result.as_of_timestamp == df.index[-1].isoformat()

    with connect() as conn:
        orders = repo.list_orders_for_deployment(conn, dep_id)
        dep = repo.get_deployment(conn, dep_id)
    assert len(orders) == 1
    assert orders[0]["status"] == "SUBMITTED"
    assert dep["last_run_at"] is not None


def test_run_deployment_is_idempotent_for_same_bar(fresh_db, monkeypatch):
    """Firing twice for the same trading day must not place two orders.
    Same last bar -> same as_of -> same idempotency hash -> the second
    submission is SKIPPED_DUPLICATE."""
    dep_id = _seed_deployment(fresh_db, armed=True)

    df = make_synthetic_ohlcv(n=300).copy()
    df.loc[df.index[-1], "close"] = 200.0
    monkeypatch.setattr(scheduler, "get_prices", _recording_prices(df))
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(scheduler, "account_fx_factor", lambda t: 1.0)
    # Portfolio stays empty across both fires — isolates the hash mechanism
    # from the "portfolio already caught up" path.
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])
    monkeypatch.setattr(t212_client, "get_account_cash", lambda: {"free": 1_000_000.0})
    # The 2nd run reconciles the 1st run's order — keep it off the network.
    monkeypatch.setattr(t212_client, "get_order", lambda oid: {"status": "NEW"})

    calls = []

    def fake_place(ticker, quantity):
        calls.append((ticker, quantity))
        return _T212_RESPONSE

    monkeypatch.setattr(t212_client, "place_market_order", fake_place)

    first = scheduler.run_deployment(dep_id)
    second = scheduler.run_deployment(dep_id)

    assert first.submit_results[0].status == "SUBMITTED"
    assert second.submit_results[0].status == "SKIPPED_DUPLICATE"
    assert len(calls) == 1, "T212 must be hit once across two fires of one bar"
    with connect() as conn:
        assert len(repo.list_orders_for_deployment(conn, dep_id)) == 1


def test_run_deployment_flat_signal_places_no_orders(fresh_db, monkeypatch):
    dep_id = _seed_deployment(fresh_db, armed=True, strategy_code=_FLAT_STRATEGY)

    monkeypatch.setattr(scheduler, "get_prices",
                        _recording_prices(make_synthetic_ohlcv(n=300)))
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(scheduler, "account_fx_factor", lambda t: 1.0)
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])

    placed = []
    monkeypatch.setattr(t212_client, "place_market_order",
                        lambda t, q: placed.append((t, q)))

    result = scheduler.run_deployment(dep_id)

    assert result.status == "OK"
    assert result.intent_count == 0
    assert result.submit_results == []
    assert placed == []


# --- run_deployment: guard rails -----------------------------------------


def test_run_deployment_skips_disarmed(fresh_db, monkeypatch):
    dep_id = _seed_deployment(fresh_db, armed=False)

    fake_prices = _recording_prices(make_synthetic_ohlcv(n=10))
    monkeypatch.setattr(scheduler, "get_prices", fake_prices)
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")

    result = scheduler.run_deployment(dep_id)

    assert result.status == "DISARMED"
    assert fake_prices.calls == [], "a disarmed deployment must not fetch data"
    with connect() as conn:
        assert repo.get_deployment(conn, dep_id)["last_run_at"] is None


def test_run_deployment_unknown_id_returns_not_found(fresh_db):
    result = scheduler.run_deployment("no-such-deployment")
    assert result.status == "NOT_FOUND"
    assert result.submit_results == []


def test_run_deployment_empty_price_data_returns_no_data(fresh_db, monkeypatch):
    dep_id = _seed_deployment(fresh_db, armed=True)
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    monkeypatch.setattr(scheduler, "get_prices", lambda *a, **k: empty)
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")

    result = scheduler.run_deployment(dep_id)

    assert result.status == "NO_DATA"
    assert result.intent_count == 0
    # An attempt was made, so last_run_at is stamped.
    with connect() as conn:
        assert repo.get_deployment(conn, dep_id)["last_run_at"] is not None


def test_run_deployment_catches_exceptions(fresh_db, monkeypatch):
    """A failure mid-pipeline becomes an ERROR result, never an uncaught
    exception (which would vanish inside the scheduler thread)."""
    dep_id = _seed_deployment(fresh_db, armed=True)

    def boom(*args, **kwargs):
        raise RuntimeError("yfinance exploded")

    monkeypatch.setattr(scheduler, "get_prices", boom)
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")

    result = scheduler.run_deployment(dep_id)

    assert result.status == "ERROR"
    assert "yfinance exploded" in (result.error or "")
    with connect() as conn:
        assert repo.get_deployment(conn, dep_id)["last_run_at"] is not None


# --- cash precheck (Phase 14) --------------------------------------------


def test_run_deployment_insufficient_cash_skips_submission(fresh_db, monkeypatch):
    """When T212 free cash is below the cash needed for the BUY intents,
    the run is marked INSUFFICIENT_CASH and no order is placed — the
    deployment stays armed so the next bar gets a fresh check."""
    dep_id = _seed_deployment(fresh_db, armed=True)

    df = make_synthetic_ohlcv(n=300).copy()
    df.loc[df.index[-1], "close"] = 200.0  # BUY qty 12.5, cash needed $2500
    monkeypatch.setattr(scheduler, "get_prices", _recording_prices(df))
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(scheduler, "account_fx_factor", lambda t: 1.0)
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])
    # Free cash well below $2500 — precheck should trip.
    monkeypatch.setattr(t212_client, "get_account_cash", lambda: {"free": 100.0})

    placed = []
    monkeypatch.setattr(t212_client, "place_market_order",
                        lambda t, q: placed.append((t, q)) or {"id": 1})

    result = scheduler.run_deployment(dep_id)

    assert result.status == "INSUFFICIENT_CASH"
    assert placed == [], "no order may be submitted when cash is short"
    assert result.error and "need" in result.error and "free" in result.error
    # Deployment stays armed so the next bar gets another chance.
    with connect() as conn:
        assert repo.get_deployment(conn, dep_id)["armed"] == 1


def test_run_deployment_cash_precheck_failure_does_not_block(fresh_db, monkeypatch):
    """The precheck is best-effort — a failing get_account_cash call
    (network/auth error) is swallowed and the order is submitted normally.
    Without this, every transient T212 hiccup would block the deployment."""
    dep_id = _seed_deployment(fresh_db, armed=True)

    df = make_synthetic_ohlcv(n=300).copy()
    df.loc[df.index[-1], "close"] = 200.0
    monkeypatch.setattr(scheduler, "get_prices", _recording_prices(df))
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(scheduler, "account_fx_factor", lambda t: 1.0)
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])

    def boom():
        raise RuntimeError("T212 hiccup")
    monkeypatch.setattr(t212_client, "get_account_cash", boom)

    placed = []
    monkeypatch.setattr(t212_client, "place_market_order",
                        lambda t, q: placed.append((t, q)) or _T212_RESPONSE)

    result = scheduler.run_deployment(dep_id)

    assert result.status == "OK"
    assert placed == [("AAPL_US_EQ", 12.5)]


# --- resolve_price_ticker -------------------------------------------------


def test_resolve_price_ticker_uses_t212_shortname(monkeypatch):
    monkeypatch.setattr(
        instruments, "find_by_ticker",
        lambda q: [{"ticker": "AAPL_US_EQ", "shortName": "AAPL"}],
    )
    assert scheduler.resolve_price_ticker("AAPL_US_EQ") == "AAPL"


def test_resolve_price_ticker_strips_suffix_when_not_found(monkeypatch):
    monkeypatch.setattr(instruments, "find_by_ticker", lambda q: [])
    assert scheduler.resolve_price_ticker("MSFT_US_EQ") == "MSFT"


def test_resolve_price_ticker_passes_plain_symbol_through(monkeypatch):
    monkeypatch.setattr(instruments, "find_by_ticker", lambda q: [])
    assert scheduler.resolve_price_ticker("SPY") == "SPY"


def test_resolve_price_ticker_survives_lookup_failure(monkeypatch):
    def boom(q):
        raise t212_client.T212UpstreamError("instruments endpoint down")

    monkeypatch.setattr(instruments, "find_by_ticker", boom)
    # Falls back to the suffix strip; must not raise.
    assert scheduler.resolve_price_ticker("AAPL_US_EQ") == "AAPL"


# --- registration helpers -------------------------------------------------


def test_register_deployment_adds_cron_job():
    scheduler.register_deployment({"id": "dep-1", "schedule_cron": "5 21 * * 1-5"})
    job = scheduler.get_scheduler().get_job("deployment:dep-1")
    assert job is not None
    assert isinstance(job.trigger, CronTrigger)
    assert tuple(job.args) == ("dep-1",)


def test_unregister_deployment_removes_job():
    scheduler.register_deployment({"id": "dep-1", "schedule_cron": "5 21 * * 1-5"})
    scheduler.unregister_deployment("dep-1")
    assert scheduler.get_scheduler().get_job("deployment:dep-1") is None


def test_unregister_missing_deployment_is_a_noop():
    scheduler.unregister_deployment("never-registered")  # must not raise


def test_register_deployment_replaces_existing():
    scheduler.register_deployment({"id": "dep-1", "schedule_cron": "5 21 * * 1-5"})
    scheduler.register_deployment({"id": "dep-1", "schedule_cron": "30 14 * * *"})
    jobs = scheduler.get_scheduler().get_jobs()
    assert len(jobs) == 1, "re-registering must update in place, not duplicate"
    assert "minute='30'" in str(jobs[0].trigger)


def test_register_deployment_rejects_bad_cron():
    with pytest.raises(ValueError):
        scheduler.register_deployment({"id": "dep-x", "schedule_cron": "not-a-cron"})


def test_register_all_armed_registers_only_armed(fresh_db):
    _seed_deployment(fresh_db, deployment_id="dep-armed-1", armed=True)
    _seed_deployment(fresh_db, deployment_id="dep-armed-2", armed=True)
    _seed_deployment(fresh_db, deployment_id="dep-off", armed=False)

    scheduler.register_all_armed()

    sched = scheduler.get_scheduler()
    assert sched.get_job("deployment:dep-armed-1") is not None
    assert sched.get_job("deployment:dep-armed-2") is not None
    assert sched.get_job("deployment:dep-off") is None
    assert len(sched.get_jobs()) == 2


def test_register_all_armed_skips_a_bad_cron(fresh_db):
    """One deployment with a broken crontab must not block the others."""
    _seed_deployment(fresh_db, deployment_id="dep-good", armed=True,
                     cron="5 21 * * 1-5")
    _seed_deployment(fresh_db, deployment_id="dep-bad", armed=True,
                     cron="nonsense")

    scheduler.register_all_armed()  # must not raise

    sched = scheduler.get_scheduler()
    assert sched.get_job("deployment:dep-good") is not None
    assert sched.get_job("deployment:dep-bad") is None


def test_start_runs_scheduler_and_registers_armed(fresh_db, monkeypatch):
    _seed_deployment(fresh_db, deployment_id="dep-armed", armed=True)
    # start() also runs catch-up, which would otherwise do a real network
    # run for this never-run deployment — stub it out.
    monkeypatch.setattr(scheduler, "catch_up_missed_runs", lambda: None)

    scheduler.start()

    sched = scheduler.get_scheduler()
    assert sched.running
    assert sched.get_job("deployment:dep-armed") is not None
    # The _fresh_scheduler fixture shuts the thread down in teardown.


# --- validate_cron + lookback_days ---------------------------------------


def test_validate_cron_accepts_a_valid_crontab():
    scheduler.validate_cron("5 21 * * 1-5")  # must not raise


def test_validate_cron_rejects_garbage():
    with pytest.raises(ValueError):
        scheduler.validate_cron("not a valid cron")


def test_lookback_days_per_timeframe():
    assert scheduler.lookback_days("1d") == 800
    assert scheduler.lookback_days("1h") == 60
    assert scheduler.lookback_days("5m") == 30
    assert scheduler.lookback_days("weird") == 800  # default fallback


# --- account_fx_factor (Phase 11.9 — currency-correct sizing) -------------


def test_account_fx_factor_same_currency_is_one(monkeypatch):
    monkeypatch.setattr(t212_client, "get_account_metadata",
                        lambda: {"currencyCode": "USD"})
    monkeypatch.setattr(instruments, "find_by_ticker",
                        lambda t: [{"ticker": t, "currencyCode": "USD"}])
    fx_prices = _recording_prices(pd.DataFrame({"close": [0.8]}))
    monkeypatch.setattr(scheduler, "get_prices", fx_prices)

    assert scheduler.account_fx_factor("AAPL_US_EQ") == 1.0
    assert fx_prices.calls == [], "no FX fetch when currencies already match"


def test_account_fx_factor_different_currency_fetches_rate(monkeypatch):
    monkeypatch.setattr(t212_client, "get_account_metadata",
                        lambda: {"currencyCode": "GBP"})
    monkeypatch.setattr(instruments, "find_by_ticker",
                        lambda t: [{"ticker": t, "currencyCode": "USD"}])
    fx_prices = _recording_prices(pd.DataFrame({"close": [0.81, 0.79, 0.8]}))
    monkeypatch.setattr(scheduler, "get_prices", fx_prices)

    assert scheduler.account_fx_factor("AAPL_US_EQ") == 0.8
    # fetched the instrument->account pair; the latest close is the rate.
    assert fx_prices.calls[0][0] == "USDGBP=X"


def test_account_fx_factor_instrument_not_found_raises(monkeypatch):
    monkeypatch.setattr(t212_client, "get_account_metadata",
                        lambda: {"currencyCode": "GBP"})
    monkeypatch.setattr(instruments, "find_by_ticker", lambda t: [])
    with pytest.raises(RuntimeError):
        scheduler.account_fx_factor("WAT_XX_EQ")


def test_account_fx_factor_missing_fx_data_raises(monkeypatch):
    monkeypatch.setattr(t212_client, "get_account_metadata",
                        lambda: {"currencyCode": "GBP"})
    monkeypatch.setattr(instruments, "find_by_ticker",
                        lambda t: [{"ticker": t, "currencyCode": "USD"}])
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    monkeypatch.setattr(scheduler, "get_prices", lambda *a, **k: empty)
    with pytest.raises(RuntimeError):
        scheduler.account_fx_factor("AAPL_US_EQ")


def test_run_deployment_applies_fx_factor(fresh_db, monkeypatch):
    """The FX factor flows through run_deployment into sizing: a 0.5
    factor doubles the share count."""
    dep_id = _seed_deployment(fresh_db, armed=True)
    df = make_synthetic_ohlcv(n=300).copy()
    df.loc[df.index[-1], "close"] = 200.0
    monkeypatch.setattr(scheduler, "get_prices", _recording_prices(df))
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(scheduler, "account_fx_factor", lambda t: 0.5)
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])

    placed = []
    monkeypatch.setattr(t212_client, "place_market_order",
                        lambda tk, q: placed.append((tk, q)) or _T212_RESPONSE)

    result = scheduler.run_deployment(dep_id)

    assert result.status == "OK"
    # 0.5 signal * 5000 capital = 2500; effective price 200*0.5=100 -> 25
    assert placed == [("AAPL_US_EQ", 25.0)]


# --- missed-run catch-up (Phase 12.2) ------------------------------------


def test_run_is_due_when_never_run():
    now = datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc)
    assert scheduler._run_is_due("0 12 * * *", None, now) is True


def test_run_is_due_when_a_scheduled_fire_was_missed():
    # Noon-daily cron; last ran two days ago -> the 19th and 20th noon
    # fires fell in (last_run, now], so a run is overdue.
    last_run = "2026-05-18T12:00:05+00:00"
    now = datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc)
    assert scheduler._run_is_due("0 12 * * *", last_run, now) is True


def test_run_is_due_false_when_up_to_date():
    # Ran just after today's noon fire; the next fire (tomorrow noon) is
    # still in the future -> not due.
    last_run = "2026-05-20T12:00:05+00:00"
    now = datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc)
    assert scheduler._run_is_due("0 12 * * *", last_run, now) is False


def test_catch_up_runs_a_due_deployment(fresh_db, monkeypatch):
    _seed_deployment(fresh_db, deployment_id="dep-due", armed=True)  # never run -> due
    ran = []
    monkeypatch.setattr(scheduler, "run_deployment", lambda i: ran.append(i))
    scheduler.catch_up_missed_runs()
    assert ran == ["dep-due"]


def test_catch_up_skips_an_up_to_date_deployment(fresh_db, monkeypatch):
    dep_id = _seed_deployment(fresh_db, deployment_id="dep-fresh", armed=True)
    with connect() as conn:
        repo.update_deployment_last_run(conn, dep_id, _now())  # ran just now
    ran = []
    monkeypatch.setattr(scheduler, "run_deployment", lambda i: ran.append(i))
    scheduler.catch_up_missed_runs()
    assert ran == []


# --- circuit breaker (Phase 12.4) ----------------------------------------


def test_circuit_breaker_off_when_no_limit():
    dep = {"ticker": "AAPL_US_EQ", "max_loss_value": None}
    portfolio = [{"ticker": "AAPL_US_EQ", "ppl": -9999.0}]
    assert scheduler._circuit_breaker_tripped(dep, portfolio) is False


def test_circuit_breaker_trips_when_loss_exceeds_limit():
    dep = {"ticker": "AAPL_US_EQ", "max_loss_value": 500.0}
    portfolio = [{"ticker": "AAPL_US_EQ", "ppl": -600.0}]
    assert scheduler._circuit_breaker_tripped(dep, portfolio) is True


def test_circuit_breaker_holds_when_loss_within_limit():
    dep = {"ticker": "AAPL_US_EQ", "max_loss_value": 500.0}
    portfolio = [{"ticker": "AAPL_US_EQ", "ppl": -200.0}]
    assert scheduler._circuit_breaker_tripped(dep, portfolio) is False


def test_circuit_breaker_off_when_no_position_held():
    dep = {"ticker": "AAPL_US_EQ", "max_loss_value": 500.0}
    assert scheduler._circuit_breaker_tripped(dep, []) is False


def test_run_deployment_trips_circuit_breaker_and_disarms(fresh_db, monkeypatch):
    dep_id = _seed_deployment(fresh_db, armed=True, max_loss_value=500.0)
    monkeypatch.setattr(scheduler, "get_prices",
                        _recording_prices(make_synthetic_ohlcv(n=300)))
    monkeypatch.setattr(scheduler, "resolve_price_ticker", lambda t: "AAPL")
    # The deployment's AAPL position is down 600 — past the 500 limit.
    monkeypatch.setattr(t212_client, "get_portfolio",
                        lambda: [{"ticker": "AAPL_US_EQ", "quantity": 5.0, "ppl": -600.0}])
    placed = []
    monkeypatch.setattr(t212_client, "place_market_order",
                        lambda tk, q: placed.append((tk, q)))

    result = scheduler.run_deployment(dep_id)

    assert result.status == "CIRCUIT_BREAKER"
    assert placed == [], "a tripped breaker places nothing"
    with connect() as conn:
        assert repo.get_deployment(conn, dep_id)["armed"] == 0  # auto-disarmed


# --- hardening (Phase 12.8) ----------------------------------------------


def test_run_deployment_errors_on_an_inactive_strategy(fresh_db):
    """A deployment whose strategy has been soft-deleted must not run it."""
    dep_id = _seed_deployment(fresh_db, armed=True)
    with connect() as conn:
        repo.soft_delete_strategy(conn, f"strat-{dep_id}")
    result = scheduler.run_deployment(dep_id)
    assert result.status == "ERROR"
    assert "inactive" in (result.error or "").lower()
