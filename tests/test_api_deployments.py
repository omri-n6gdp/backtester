"""Phase 11.6 — tests for the /deployments HTTP routes.

Covers create / list / get / arm / disarm / run / tracking. The routes
that reach the network (the manual /run, and /tracking's expected
backtest) have their price + T212 seams monkeypatched — no live calls.
"""
from __future__ import annotations

from backend.t212 import client as t212_client
from tests.conftest import install_fake_prices, make_synthetic_ohlcv, post_strategy


# Always-50%-long strategy: deterministic, valid, no warm-up.
_STRATEGY = """import pandas as pd

def strategy(data, params):
    return pd.Series(0.5, index=data.index)
"""


def _create_deployment(client, strategy_id, **overrides):
    """POST /deployments with sensible defaults. Returns the raw response."""
    body = {
        "strategy_id": strategy_id,
        "ticker": "AAPL_US_EQ",
        "timeframe": "1d",
        "params": {},
        "deployment_capital": 5000.0,
        "order_type": "market",
        "schedule_cron": "5 21 * * 1-5",
        "confirmation_token": "ARM",
    }
    body.update(overrides)
    return client.post("/deployments", json=body)


# --- create ---------------------------------------------------------------


def test_create_deployment_happy_path(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    resp = _create_deployment(api_client, sid)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["strategy_id"] == sid
    assert body["ticker"] == "AAPL_US_EQ"
    assert body["armed"] is True
    assert body["deployment_capital"] == 5000.0
    assert body["last_run_at"] is None


def test_create_deployment_requires_arm_token(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    resp = _create_deployment(api_client, sid, confirmation_token="yes please")
    assert resp.status_code == 403


def test_create_deployment_unknown_strategy_404(api_client):
    resp = _create_deployment(api_client, "no-such-strategy")
    assert resp.status_code == 404


def test_create_deployment_bad_cron_400(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    resp = _create_deployment(api_client, sid, schedule_cron="not a cron")
    assert resp.status_code == 400


def test_create_deployment_bad_order_type_400(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    resp = _create_deployment(api_client, sid, order_type="stop")
    assert resp.status_code == 400


# --- list / get -----------------------------------------------------------


def test_list_deployments(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    _create_deployment(api_client, sid)
    _create_deployment(api_client, sid, ticker="MSFT_US_EQ")
    resp = api_client.get("/deployments")
    assert resp.status_code == 200
    assert len(resp.json()["deployments"]) == 2


def test_list_deployments_filter_by_strategy(api_client):
    sid_a = post_strategy(api_client, "A", _STRATEGY)
    sid_b = post_strategy(api_client, "B", _STRATEGY)
    _create_deployment(api_client, sid_a)
    _create_deployment(api_client, sid_b)
    resp = api_client.get("/deployments", params={"strategy_id": sid_a})
    deployments = resp.json()["deployments"]
    assert len(deployments) == 1
    assert deployments[0]["strategy_id"] == sid_a


def test_get_deployment(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    dep = _create_deployment(api_client, sid).json()
    resp = api_client.get(f"/deployments/{dep['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == dep["id"]


def test_get_deployment_404(api_client):
    assert api_client.get("/deployments/does-not-exist").status_code == 404


# --- arm / disarm ---------------------------------------------------------


def test_disarm_deployment(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    dep = _create_deployment(api_client, sid).json()
    resp = api_client.post(f"/deployments/{dep['id']}/disarm")
    assert resp.status_code == 200
    assert resp.json()["armed"] is False


def test_arm_deployment_requires_token(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    dep = _create_deployment(api_client, sid).json()
    api_client.post(f"/deployments/{dep['id']}/disarm")

    bad = api_client.post(
        f"/deployments/{dep['id']}/arm", json={"confirmation_token": "nope"},
    )
    assert bad.status_code == 403

    ok = api_client.post(
        f"/deployments/{dep['id']}/arm", json={"confirmation_token": "ARM"},
    )
    assert ok.status_code == 200
    assert ok.json()["armed"] is True


def test_arm_unknown_deployment_404(api_client):
    resp = api_client.post(
        "/deployments/nope/arm", json={"confirmation_token": "ARM"},
    )
    assert resp.status_code == 404


def test_disarm_unknown_deployment_404(api_client):
    assert api_client.post("/deployments/nope/disarm").status_code == 404


# --- manual run -----------------------------------------------------------


def test_run_deployment_places_order(api_client, monkeypatch):
    sid = post_strategy(api_client, "S", _STRATEGY)
    dep = _create_deployment(api_client, sid).json()

    df = make_synthetic_ohlcv(n=300).copy()
    df.loc[df.index[-1], "close"] = 200.0
    monkeypatch.setattr("backend.deploy.scheduler.get_prices", lambda *a, **k: df)
    monkeypatch.setattr("backend.deploy.scheduler.resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])
    monkeypatch.setattr("backend.deploy.scheduler.account_fx_factor", lambda t: 1.0)

    placed = []

    def fake_place(ticker, quantity):
        placed.append((ticker, quantity))
        return {"id": 12345}

    monkeypatch.setattr(t212_client, "place_market_order", fake_place)

    resp = api_client.post(f"/deployments/{dep['id']}/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "OK"
    assert body["intent_count"] == 1
    assert body["submitted"] == 1
    assert body["results"][0]["status"] == "SUBMITTED"
    # 0.5 signal * $5000 / $200 = 12.5 shares BUY of the T212 instrument code.
    assert placed == [("AAPL_US_EQ", 12.5)]


def test_run_disarmed_deployment_reports_disarmed(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    dep = _create_deployment(api_client, sid).json()
    api_client.post(f"/deployments/{dep['id']}/disarm")
    resp = api_client.post(f"/deployments/{dep['id']}/run")
    assert resp.status_code == 200
    assert resp.json()["status"] == "DISARMED"


def test_run_unknown_deployment_404(api_client):
    assert api_client.post("/deployments/nope/run").status_code == 404


# --- tracking -------------------------------------------------------------


def test_tracking_returns_expected_curve(api_client, monkeypatch):
    sid = post_strategy(api_client, "S", _STRATEGY)
    dep = _create_deployment(api_client, sid).json()

    install_fake_prices(monkeypatch, lambda *a, **k: make_synthetic_ohlcv(n=300))
    monkeypatch.setattr("backend.deploy.scheduler.resolve_price_ticker", lambda t: "AAPL")

    resp = api_client.get(f"/deployments/{dep['id']}/tracking")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deployment"]["id"] == dep["id"]
    assert body["orders"] == []
    assert body["expected_error"] is None
    assert len(body["expected_equity"]) == 300
    assert len(body["expected_timestamps"]) == 300


def test_tracking_includes_orders_after_a_run(api_client, monkeypatch):
    sid = post_strategy(api_client, "S", _STRATEGY)
    dep = _create_deployment(api_client, sid).json()

    df = make_synthetic_ohlcv(n=300).copy()
    df.loc[df.index[-1], "close"] = 200.0
    monkeypatch.setattr("backend.deploy.scheduler.get_prices", lambda *a, **k: df)
    monkeypatch.setattr("backend.api.routes.get_prices", lambda *a, **k: df)
    monkeypatch.setattr("backend.deploy.scheduler.resolve_price_ticker", lambda t: "AAPL")
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])
    monkeypatch.setattr("backend.deploy.scheduler.account_fx_factor", lambda t: 1.0)
    monkeypatch.setattr(t212_client, "place_market_order", lambda tk, q: {"id": 7})

    api_client.post(f"/deployments/{dep['id']}/run")

    body = api_client.get(f"/deployments/{dep['id']}/tracking").json()
    assert len(body["orders"]) == 1
    assert body["orders"][0]["status"] == "SUBMITTED"
    assert body["orders"][0]["side"] == "BUY"


def test_tracking_unknown_deployment_404(api_client):
    assert api_client.get("/deployments/nope/tracking").status_code == 404


# --- max position size (Phase 12.3) --------------------------------------


def test_create_deployment_with_max_position_value(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    resp = _create_deployment(api_client, sid, max_position_value=2500.0)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["max_position_value"] == 2500.0
    # round-trips on GET
    got = api_client.get(f"/deployments/{body['id']}").json()
    assert got["max_position_value"] == 2500.0


def test_create_deployment_without_max_position_value_is_uncapped(api_client):
    sid = post_strategy(api_client, "S", _STRATEGY)
    body = _create_deployment(api_client, sid).json()
    assert body["max_position_value"] is None
