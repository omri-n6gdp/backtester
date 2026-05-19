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
