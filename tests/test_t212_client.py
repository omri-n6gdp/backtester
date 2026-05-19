"""Unit tests for backend.t212.client.

All tests mock the httpx transport — no live calls to Trading 212.
The mock layer captures the outgoing Authorization header so the
Basic(key, secret) contract is locked in as a regression test.
"""
from __future__ import annotations

import base64
import json
from typing import Callable, List

import httpx
import pytest

from backend.t212 import client as t212


def _make_mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Build an httpx.Client whose transport is the provided handler.

    Mirrors the real client's auth + base_url so the auth header that
    reaches the handler is the same one the production client would send.
    """
    from backend.config import settings

    auth = httpx.BasicAuth(settings.t212_demo_key, settings.t212_demo_secret)
    transport = httpx.MockTransport(handler)
    return httpx.Client(
        base_url=t212.BASE_URL_DEMO,
        auth=auth,
        timeout=t212.DEFAULT_TIMEOUT,
        transport=transport,
    )


@pytest.fixture
def captured_requests():
    """Yield (list, install) where `list` accumulates every httpx.Request
    sent through the mock transport. Tests assert on its contents.

    Test must call `install(handler)` to wire up the mock client and
    register the override. Teardown clears the override.
    """
    captured: List[httpx.Request] = []

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        def wrapping_handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return handler(request)

        mock_client = _make_mock_client(wrapping_handler)
        t212._set_test_client(mock_client)

    yield captured, install

    t212._set_test_client(None)


# --- Auth contract --------------------------------------------------------


def test_outgoing_authorization_header_is_basic_key_colon_secret(captured_requests):
    """The load-bearing learning from the Phase 10 probe: T212 v0 uses
    Basic auth with key:secret, not a bare key. Pin this behavior."""
    captured, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"free": 5000.0, "total": 5000.0})

    install(handler)

    t212.get_account_cash()

    assert len(captured) == 1
    auth = captured[0].headers.get("authorization")
    assert auth is not None, "no Authorization header on outgoing request"
    assert auth.startswith("Basic "), f"expected Basic scheme, got {auth.split(' ', 1)[0]!r}"

    from backend.config import settings
    decoded = base64.b64decode(auth.removeprefix("Basic ").encode()).decode()
    assert decoded == f"{settings.t212_demo_key}:{settings.t212_demo_secret}"


# --- Happy paths ----------------------------------------------------------


def test_get_account_cash_returns_parsed_dict(captured_requests):
    captured, install = captured_requests
    payload = {
        "free": 5000.0, "total": 5000.0, "ppl": 0.0,
        "result": 0.0, "invested": 0.0, "pieCash": 0.0, "blocked": 0.0,
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v0/equity/account/cash"
        return httpx.Response(200, json=payload)

    install(handler)
    assert t212.get_account_cash() == payload


def test_get_portfolio_returns_list(captured_requests):
    captured, install = captured_requests
    payload = [
        {"ticker": "AAPL_US_EQ", "quantity": 1.5, "averagePrice": 150.0,
         "currentPrice": 180.0, "ppl": 45.0},
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v0/equity/portfolio"
        return httpx.Response(200, json=payload)

    install(handler)
    assert t212.get_portfolio() == payload


def test_get_portfolio_empty_account_returns_empty_list(captured_requests):
    """T212 returns [] for an account with no positions; we pass it through."""
    captured, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    install(handler)
    assert t212.get_portfolio() == []


def test_get_orders_handles_null_body_as_empty_list(captured_requests):
    """Defensive: if T212 returns 204/empty body, the list helpers yield []."""
    captured, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    install(handler)
    assert t212.get_orders() == []


def test_get_instruments_passes_through_large_list(captured_requests):
    captured, install = captured_requests
    payload = [
        {"ticker": f"FAKE{i}_US_EQ", "name": f"Fake Co {i}", "type": "STOCK"}
        for i in range(5)
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v0/equity/metadata/instruments"
        return httpx.Response(200, json=payload)

    install(handler)
    assert t212.get_instruments() == payload


def test_get_pies_returns_list(captured_requests):
    captured, install = captured_requests

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v0/equity/pies"
        return httpx.Response(200, json=[])

    install(handler)
    assert t212.get_pies() == []


# --- Error mapping --------------------------------------------------------


def test_401_raises_auth_error(captured_requests):
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        # T212 sends text/plain with a 1-byte body for 401s — emulate that.
        return httpx.Response(401, text="\n", headers={"content-type": "text/plain"})

    install(handler)
    with pytest.raises(t212.T212AuthError) as exc_info:
        t212.get_account_cash()
    assert "401" in str(exc_info.value)
    assert "T212_DEMO_SECRET" in str(exc_info.value), \
        "auth error message should call out both .env names"


def test_429_raises_rate_limited_with_retry_after(captured_requests):
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "12"}, text="rate limited")

    install(handler)
    with pytest.raises(t212.T212RateLimited) as exc_info:
        t212.get_account_cash()
    assert exc_info.value.retry_after == 12.0


def test_429_without_retry_after_still_raises(captured_requests):
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="")

    install(handler)
    with pytest.raises(t212.T212RateLimited) as exc_info:
        t212.get_account_cash()
    assert exc_info.value.retry_after is None


def test_500_raises_upstream_error(captured_requests):
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    install(handler)
    with pytest.raises(t212.T212UpstreamError) as exc_info:
        t212.get_account_cash()
    assert "503" in str(exc_info.value)


def test_other_4xx_raises_generic_error(captured_requests):
    """A 4xx that isn't 401/429 should still surface as T212Error."""
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    install(handler)
    with pytest.raises(t212.T212Error) as exc_info:
        t212.get_account_cash()
    # Subclasses are also T212Error; check this one isn't an auth/rate/upstream subclass.
    assert not isinstance(exc_info.value, t212.T212AuthError)
    assert not isinstance(exc_info.value, t212.T212RateLimited)
    assert not isinstance(exc_info.value, t212.T212UpstreamError)
    assert "403" in str(exc_info.value)


def test_timeout_raises_transport_error(captured_requests):
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated")

    install(handler)
    with pytest.raises(t212.T212TransportError):
        t212.get_account_cash()


def test_network_error_raises_transport_error(captured_requests):
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated DNS failure")

    install(handler)
    with pytest.raises(t212.T212TransportError):
        t212.get_account_cash()


def test_success_with_non_json_body_raises_upstream(captured_requests):
    """If T212 misbehaves and returns 200 with garbage body, surface a clear error."""
    _, install = captured_requests

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json", headers={"content-type": "text/plain"})

    install(handler)
    with pytest.raises(t212.T212UpstreamError):
        t212.get_account_cash()


# --- Order write endpoints (Phase 11.1) -----------------------------------


_PLACE_LIMIT_RESPONSE = {
    "id": 49150662798,
    "strategy": "QUANTITY",
    "type": "LIMIT",
    "ticker": "AAPL_US_EQ",
    "quantity": 1.0,
    "filledQuantity": 0,
    "limitPrice": 10.0,
    "status": "NEW",
    "currency": "GBP",
    "extendedHours": False,
    "initiatedFrom": "API",
    "side": "BUY",
    "timeInForce": "DAY",
    "createdAt": "2026-05-19T15:04:14.847+03:00",
    "instrument": {
        "ticker": "AAPL_US_EQ",
        "name": "Apple",
        "isin": "US0378331005",
        "currency": "USD",
    },
}


def test_place_limit_order_sends_correct_body_and_returns_response(captured_requests):
    captured, install = captured_requests

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/api/v0/equity/orders/limit"
        body = json.loads(req.content)
        assert body == {
            "ticker": "AAPL_US_EQ",
            "quantity": 1.0,
            "limitPrice": 10.0,
            "timeValidity": "DAY",
        }
        return httpx.Response(200, json=_PLACE_LIMIT_RESPONSE)

    install(handler)
    resp = t212.place_limit_order("AAPL_US_EQ", 1.0, 10.0)
    assert resp == _PLACE_LIMIT_RESPONSE
    assert resp["id"] == 49150662798
    assert resp["status"] == "NEW"


def test_place_limit_order_respects_time_validity_override(captured_requests):
    captured, install = captured_requests

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["timeValidity"] == "GTC"
        return httpx.Response(200, json=_PLACE_LIMIT_RESPONSE)

    install(handler)
    t212.place_limit_order("AAPL_US_EQ", 1.0, 10.0, time_validity="GTC")


def test_place_limit_order_400_surfaces_t212_detail(captured_requests):
    """T212 returns RFC-7807 problem details on errors. The detail/title
    must reach the caller via the exception message."""
    _, install = captured_requests
    err_body = {
        "type": "/api-errors/business/insufficient-funds",
        "title": "Insufficient funds",
        "status": 400,
        "detail": "Not enough free cash to place this order",
        "traceId": "abc123",
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=err_body)

    install(handler)
    with pytest.raises(t212.T212Error) as exc_info:
        t212.place_limit_order("AAPL_US_EQ", 1000.0, 200.0)
    msg = str(exc_info.value)
    assert "400" in msg
    assert "Insufficient funds" in msg or "Not enough free cash" in msg


def test_place_market_order_sends_correct_body(captured_requests):
    captured, install = captured_requests
    market_response = {**_PLACE_LIMIT_RESPONSE, "type": "MARKET"}
    market_response.pop("limitPrice", None)

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/api/v0/equity/orders/market"
        body = json.loads(req.content)
        # Market orders take only ticker + quantity — NO limitPrice, NO timeValidity.
        assert body == {"ticker": "AAPL_US_EQ", "quantity": 1.0}
        return httpx.Response(200, json=market_response)

    install(handler)
    resp = t212.place_market_order("AAPL_US_EQ", 1.0)
    assert resp["type"] == "MARKET"


def test_place_market_order_negative_quantity_means_sell(captured_requests):
    """T212 infers side from quantity sign; the client passes it through verbatim."""
    captured, install = captured_requests

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["quantity"] == -2.5
        return httpx.Response(200, json={**_PLACE_LIMIT_RESPONSE, "side": "SELL"})

    install(handler)
    t212.place_market_order("AAPL_US_EQ", -2.5)


def test_cancel_order_returns_none_on_empty_body(captured_requests):
    """T212 sends an empty 200 body on successful cancel. Helper returns None."""
    captured, install = captured_requests

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "DELETE"
        assert req.url.path == "/api/v0/equity/orders/49150662798"
        return httpx.Response(200)

    install(handler)
    assert t212.cancel_order(49150662798) is None


def test_cancel_order_post_place_race_surfaces_as_t212error(captured_requests):
    """The probe found that DELETE immediately after place can 404 with
    "Order not found" even though the order exists. Confirm this surfaces
    as a generic T212Error (not T212AuthError / T212RateLimited / etc.)
    so callers that retry get an unambiguous signal to disambiguate via
    get_order."""
    _, install = captured_requests
    err_body = {
        "type": "/api-errors/entity-not-found",
        "title": "Requested entity not found",
        "status": 404,
        "detail": "Order not found",
        "traceId": "a6f2a86aceb337f9d082aff3001851ee",
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=err_body)

    install(handler)
    with pytest.raises(t212.T212Error) as exc_info:
        t212.cancel_order(49150662798)
    # Should be the generic T212Error — not an auth/rate/upstream subclass.
    assert not isinstance(exc_info.value, t212.T212AuthError)
    assert not isinstance(exc_info.value, t212.T212RateLimited)
    assert not isinstance(exc_info.value, t212.T212UpstreamError)
    assert not isinstance(exc_info.value, t212.T212TransportError)
    assert "404" in str(exc_info.value)


def test_get_order_returns_single_order_dict(captured_requests):
    captured, install = captured_requests

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/api/v0/equity/orders/49150662798"
        return httpx.Response(200, json=_PLACE_LIMIT_RESPONSE)

    install(handler)
    resp = t212.get_order(49150662798)
    assert resp == _PLACE_LIMIT_RESPONSE
    assert resp["id"] == 49150662798
