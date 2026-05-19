"""
Trading 212 v0 HTTP client (Demo environment, read-only in Phase 10).

Auth contract — verified live during Phase 10 setup:
    Authorization: Basic <base64(api_key:secret)>

The dated CLAUDE.md note suggested a single bare key, but a probe against
/equity/account/cash with `Authorization: <key>` returned 401 while
`Authorization: Basic base64(key:secret)` returned 200. Bare-key auth was
wrong; key:secret Basic is the actual contract.

Errors map to T212Error subclasses so route handlers can render clean
HTTP responses without inspecting status codes:
    401 -> T212AuthError
    429 -> T212RateLimited (with Retry-After if T212 sent one)
    5xx -> T212UpstreamError
    timeout / network -> T212TransportError
    other 4xx -> T212Error (generic)

The client is a module-level lru_cache singleton — same shape as
backend.llm.groq_client. Tests reset it via `_reset_client_cache()` and
inject an httpx.MockTransport via `_set_test_client()`.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx

from backend.config import settings

BASE_URL_DEMO = "https://demo.trading212.com/api/v0/"

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

logger = logging.getLogger(__name__)


# --- Errors ---------------------------------------------------------------


class T212Error(Exception):
    """Base for all Trading 212 client errors."""


class T212AuthError(T212Error):
    """HTTP 401 — bad/revoked api_key or secret."""


class T212RateLimited(T212Error):
    """HTTP 429 — back off. T212's free tier is aggressive."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class T212UpstreamError(T212Error):
    """HTTP 5xx — T212-side outage. Retry later."""


class T212TransportError(T212Error):
    """Local network failure or timeout. Not T212's fault."""


# --- Client singleton -----------------------------------------------------


@lru_cache(maxsize=1)
def _get_client() -> httpx.Client:
    """Lazy httpx.Client singleton.

    Tests reset via `_reset_client_cache()` and (optionally) inject via
    `_set_test_client(client)`. Production callers should never touch
    those seams.
    """
    auth = httpx.BasicAuth(settings.t212_demo_key, settings.t212_demo_secret)
    return httpx.Client(
        base_url=BASE_URL_DEMO,
        auth=auth,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": "backtester/0.3 (+localhost)"},
    )


def _reset_client_cache() -> None:
    """Drop the cached httpx.Client. Test seam."""
    _get_client.cache_clear()


_test_client_override: httpx.Client | None = None


def _set_test_client(client: httpx.Client | None) -> None:
    """Install / clear a test-only override. When set, _request() uses it
    instead of the lru_cache singleton. Pass None to clear."""
    global _test_client_override
    _test_client_override = client


def _current_client() -> httpx.Client:
    if _test_client_override is not None:
        return _test_client_override
    return _get_client()


# --- Request helper -------------------------------------------------------


def _request(method: str, path: str, **kwargs: Any) -> Any:
    """Issue a request; raise the right T212Error subclass on failure;
    return parsed JSON (or None for empty bodies) on success."""
    client = _current_client()

    try:
        response = client.request(method, path, **kwargs)
    except httpx.TimeoutException as exc:
        raise T212TransportError(f"T212 request timed out: {exc!s}") from exc
    except httpx.HTTPError as exc:
        raise T212TransportError(f"T212 transport error: {exc!s}") from exc

    status = response.status_code

    if 200 <= status < 300:
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise T212UpstreamError(
                f"T212 returned non-JSON success body for {method} {path}: "
                f"{response.text[:200]!r}"
            ) from exc

    # Error paths. T212 frequently returns empty / text/plain error bodies.
    body_snippet = response.text[:200] if response.content else "<empty>"

    if status == 401:
        raise T212AuthError(
            "T212 rejected credentials (HTTP 401). "
            "Verify T212_DEMO_KEY and T212_DEMO_SECRET in .env — both halves "
            "are required for Basic auth."
        )
    if status == 429:
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        suffix = f" Retry after {retry_after}s." if retry_after else ""
        raise T212RateLimited(
            f"T212 rate-limited (HTTP 429).{suffix}",
            retry_after=retry_after,
        )
    if 500 <= status < 600:
        raise T212UpstreamError(
            f"T212 returned HTTP {status} for {method} {path}: {body_snippet}"
        )

    raise T212Error(
        f"T212 returned HTTP {status} for {method} {path}: {body_snippet}"
    )


def _parse_retry_after(header_value: str | None) -> float | None:
    """T212 sends Retry-After as a delta-seconds integer when present."""
    if not header_value:
        return None
    try:
        return float(header_value)
    except ValueError:
        return None


# --- Public API -----------------------------------------------------------


def get_account_cash() -> dict[str, Any]:
    """GET /equity/account/cash → cash breakdown.

    Returns a dict with keys: free, total, ppl (open P&L), result,
    invested, pieCash, blocked. Values are floats in account currency.
    """
    return _request("GET", "equity/account/cash")


def get_account_metadata() -> dict[str, Any]:
    """GET /equity/account/info → account-level metadata (currency, id)."""
    return _request("GET", "equity/account/info")


def get_portfolio() -> list[dict[str, Any]]:
    """GET /equity/portfolio → list of open positions.

    Each entry typically has: ticker, quantity, averagePrice, currentPrice,
    ppl, fxPpl, initialFillDate. Returns [] for an empty account.
    """
    result = _request("GET", "equity/portfolio")
    return result if result is not None else []


def get_orders() -> list[dict[str, Any]]:
    """GET /equity/orders → list of currently working (open) orders."""
    result = _request("GET", "equity/orders")
    return result if result is not None else []


def get_instruments() -> list[dict[str, Any]]:
    """GET /equity/metadata/instruments → all tradable instruments.

    NOTE: this response is large (10+ MB / thousands of entries). Callers
    should prefer the cached helpers in backend.t212.instruments rather
    than calling this directly on every request.
    """
    result = _request("GET", "equity/metadata/instruments")
    return result if result is not None else []


def get_pies() -> list[dict[str, Any]]:
    """GET /equity/pies → user's pies (auto-invest portfolios)."""
    result = _request("GET", "equity/pies")
    return result if result is not None else []
