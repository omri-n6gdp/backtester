"""Tests for /t212/* read-only routes.

Mocks the client/instruments modules; no live T212 calls. Uses the
shared `api_client` fixture from tests/conftest.py.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.t212 import client as t212_client
from backend.t212 import instruments as t212_instruments


CASH_PAYLOAD = {
    "free": 5000.0, "total": 5000.0, "ppl": 0.0,
    "result": 0.0, "invested": 0.0, "pieCash": 0.0, "blocked": 0.0,
}

POSITIONS_PAYLOAD = [
    {"ticker": "AAPL_US_EQ", "quantity": 1.5, "averagePrice": 150.0,
     "currentPrice": 180.0, "ppl": 45.0, "fxPpl": 0.0},
]

ORDERS_PAYLOAD = [
    {"id": 12345, "ticker": "AAPL_US_EQ", "quantity": 1.0, "type": "LIMIT",
     "limitPrice": 200.0, "status": "WORKING"},
]

PIES_PAYLOAD = [
    {"id": 99, "name": "Tech Pie", "cash": 100.0, "dividendDetails": {}},
]

INSTRUMENTS_PAYLOAD = [
    {"ticker": "AAPL_US_EQ", "shortName": "AAPL", "type": "STOCK",
     "name": "Apple Inc.", "currencyCode": "USD", "isin": "US0378331005",
     "workingScheduleId": 56, "maxOpenQuantity": 1000.0,
     "extendedHours": True, "addedOn": "2018-01-01T00:00:00.000+00:00"},
]


@pytest.fixture
def patched_t212(monkeypatch: pytest.MonkeyPatch):
    """Stub every T212 endpoint with the default fixtures above.

    Tests can override individual stubs with monkeypatch.setattr if they
    need to exercise error paths.
    """
    monkeypatch.setattr(t212_client, "get_account_cash", lambda: CASH_PAYLOAD)
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: POSITIONS_PAYLOAD)
    monkeypatch.setattr(t212_client, "get_orders", lambda: ORDERS_PAYLOAD)
    monkeypatch.setattr(t212_client, "get_pies", lambda: PIES_PAYLOAD)
    monkeypatch.setattr(
        t212_instruments, "get_instruments",
        lambda **kw: INSTRUMENTS_PAYLOAD,
    )
    return monkeypatch


# --- Happy paths ----------------------------------------------------------


def test_account_returns_cash_dict(api_client: TestClient, patched_t212):
    resp = api_client.get("/t212/account")
    assert resp.status_code == 200
    assert resp.json() == CASH_PAYLOAD


def test_positions_returns_list(api_client: TestClient, patched_t212):
    resp = api_client.get("/t212/positions")
    assert resp.status_code == 200
    assert resp.json() == POSITIONS_PAYLOAD


def test_orders_returns_list(api_client: TestClient, patched_t212):
    resp = api_client.get("/t212/orders")
    assert resp.status_code == 200
    assert resp.json() == ORDERS_PAYLOAD


def test_pies_returns_list(api_client: TestClient, patched_t212):
    resp = api_client.get("/t212/pies")
    assert resp.status_code == 200
    assert resp.json() == PIES_PAYLOAD


def test_instruments_returns_cached_list(api_client: TestClient, patched_t212):
    resp = api_client.get("/t212/instruments")
    assert resp.status_code == 200
    assert resp.json() == INSTRUMENTS_PAYLOAD


def test_instruments_refresh_query_forwards(api_client: TestClient, monkeypatch):
    """?refresh=true must reach instruments.get_instruments(force_refresh=True)."""
    seen_kwargs: dict[str, Any] = {}

    def stub(*, force_refresh: bool = False):
        seen_kwargs["force_refresh"] = force_refresh
        return INSTRUMENTS_PAYLOAD

    monkeypatch.setattr(t212_instruments, "get_instruments", stub)
    api_client.get("/t212/instruments?refresh=true")
    assert seen_kwargs == {"force_refresh": True}

    api_client.get("/t212/instruments")
    assert seen_kwargs == {"force_refresh": False}


# --- Empty-account behavior ----------------------------------------------


def test_positions_empty_account(api_client: TestClient, monkeypatch):
    monkeypatch.setattr(t212_client, "get_portfolio", lambda: [])
    resp = api_client.get("/t212/positions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_orders_empty_account(api_client: TestClient, monkeypatch):
    monkeypatch.setattr(t212_client, "get_orders", lambda: [])
    resp = api_client.get("/t212/orders")
    assert resp.status_code == 200
    assert resp.json() == []


# --- Error mapping --------------------------------------------------------


def test_auth_error_maps_to_502(api_client: TestClient, monkeypatch):
    def raises():
        raise t212_client.T212AuthError("bad credentials")
    monkeypatch.setattr(t212_client, "get_account_cash", raises)
    resp = api_client.get("/t212/account")
    assert resp.status_code == 502
    assert "auth failed" in resp.json()["detail"].lower()


def test_rate_limited_maps_to_502_with_retry_after(api_client: TestClient, monkeypatch):
    def raises():
        raise t212_client.T212RateLimited("rate limited", retry_after=30.0)
    monkeypatch.setattr(t212_client, "get_account_cash", raises)
    resp = api_client.get("/t212/account")
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail.get("retry_after_seconds") == 30.0


def test_rate_limited_without_retry_after_omits_field(api_client: TestClient, monkeypatch):
    def raises():
        raise t212_client.T212RateLimited("rate limited", retry_after=None)
    monkeypatch.setattr(t212_client, "get_account_cash", raises)
    resp = api_client.get("/t212/account")
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "retry_after_seconds" not in detail


def test_upstream_error_maps_to_502(api_client: TestClient, monkeypatch):
    def raises():
        raise t212_client.T212UpstreamError("503 from T212")
    monkeypatch.setattr(t212_client, "get_portfolio", raises)
    resp = api_client.get("/t212/positions")
    assert resp.status_code == 502


def test_transport_error_maps_to_504(api_client: TestClient, monkeypatch):
    def raises():
        raise t212_client.T212TransportError("connection timed out")
    monkeypatch.setattr(t212_client, "get_orders", raises)
    resp = api_client.get("/t212/orders")
    assert resp.status_code == 504


def test_generic_t212_error_maps_to_502(api_client: TestClient, monkeypatch):
    def raises():
        raise t212_client.T212Error("HTTP 403: forbidden")
    monkeypatch.setattr(t212_client, "get_pies", raises)
    resp = api_client.get("/t212/pies")
    assert resp.status_code == 502


# --- 503 defense-in-depth (settings empty) -------------------------------


def test_503_when_settings_credentials_empty(api_client: TestClient, monkeypatch):
    """Even though backend.config raises hard at startup if either key is
    missing, the route includes a defense-in-depth 503 in case someone
    relaxes from_env() in the future. Patch a fake Settings with empty
    fields to exercise that branch."""
    from backend.config import Settings

    fake = Settings(
        groq_api_key="x",
        t212_demo_key="",
        t212_demo_secret="",
        t212_live_key="",
        t212_live_secret="",
    )
    monkeypatch.setattr("backend.config.settings", fake)

    resp = api_client.get("/t212/account")
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"].lower()
