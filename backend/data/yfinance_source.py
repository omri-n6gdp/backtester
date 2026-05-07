"""
Yahoo Finance fetcher (via yfinance).

Yahoo aggressively rate-limits the unofficial yfinance scraper. This module:
- enforces a global minimum interval between calls (token-bucket-ish; simple sleep)
- retries on 429 with exponential backoff (2s, 4s, 8s, 16s, 32s; max 60s)
- rotates curl_cffi browser-impersonation targets between calls (TLS-fingerprint
  rotation; strictly stronger than User-Agent rotation alone)
- normalizes the returned DataFrame to a strict schema

yfinance >=0.2.60 requires a curl_cffi session; stock requests.Session is
rejected by the library. We honour that by building a curl_cffi session per
call with a randomly-chosen impersonate target.

Public function: fetch(ticker, timeframe, start, end) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf
from curl_cffi.requests import Session as CurlSession

logger = logging.getLogger(__name__)

# --- Throttling -------------------------------------------------------------

# Global minimum interval between yfinance calls (seconds).
# Yahoo's 429 thresholds are not published; 1.5s is a defensive default.
MIN_INTERVAL_S = 1.5

_throttle_lock = threading.Lock()
_last_call_ts: float = 0.0


def _throttle() -> None:
    """Block until at least MIN_INTERVAL_S has passed since the last call."""
    global _last_call_ts
    with _throttle_lock:
        now = time.monotonic()
        wait = (_last_call_ts + MIN_INTERVAL_S) - now
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


# --- Browser impersonation rotation ----------------------------------------

# curl_cffi impersonate targets. Each value rotates the full TLS/JA3/HTTP2
# fingerprint to look like that browser version. Yahoo's bot detection keys
# off these fingerprints, not just the User-Agent.
# Pool of recent, well-supported targets in curl_cffi.
_IMPERSONATE_TARGETS = [
    "chrome124",
    "chrome131",
    "chrome133a",
    "safari17_0",
    "safari18_0",
    "edge101",
]


def _make_session() -> CurlSession:
    """Build a curl_cffi Session impersonating a randomly-chosen browser."""
    target = random.choice(_IMPERSONATE_TARGETS)
    return CurlSession(impersonate=target)


# --- Backoff ---------------------------------------------------------------

MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0


class YFinanceRateLimited(Exception):
    """Raised when yfinance hits a 429 and all retries are exhausted."""


class YFinanceFetchError(Exception):
    """Raised on any other unrecoverable yfinance failure."""


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Heuristic: yfinance surfaces 429s in a few different shapes."""
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


# --- Timeframe mapping -----------------------------------------------------

_TIMEFRAME_TO_YF = {
    "1d": "1d",
    "1h": "1h",
    "5m": "5m",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a yfinance DataFrame to the project schema.

    - Index named 'timestamp', timezone-aware UTC.
    - Columns lowercased: open, high, low, close, volume.
    - Floats. NaN-only rows dropped.
    - Sorted ascending, deduplicated on index.
    """
    if df is None or df.empty:
        return _empty_frame()

    # yfinance sometimes returns multi-level columns (e.g. for groups). Flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)

    keep = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise YFinanceFetchError(f"yfinance response missing columns: {missing}")
    df = df[keep].copy()

    # Coerce to float (volume included; fractional shares + crypto have non-int volume).
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    df = df.dropna(how="all")

    # Index: ensure DatetimeIndex, tz-aware UTC, named 'timestamp'.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {c: pd.Series(dtype=float) for c in ("open", "high", "low", "close", "volume")},
        index=idx,
    )


# --- Public API ------------------------------------------------------------

def fetch(
    ticker: str,
    timeframe: str,
    start: date,
    end: date,
    *,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance for [start, end].

    Args:
        ticker: e.g. 'SPY', 'BTC-USD', 'AAPL'.
        timeframe: one of '1d', '1h', '5m'.
        start: inclusive start date (UTC).
        end: inclusive end date (UTC).

    Returns:
        Normalized DataFrame (see _normalize). May be empty if Yahoo returns no rows.

    Raises:
        ValueError: on bad timeframe.
        YFinanceRateLimited: if all retries are exhausted on 429s.
        YFinanceFetchError: on other unrecoverable failures.
    """
    if timeframe not in _TIMEFRAME_TO_YF:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {list(_TIMEFRAME_TO_YF)}")

    interval = _TIMEFRAME_TO_YF[timeframe]
    # yfinance 'end' is exclusive, so add a day.
    end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        _throttle()
        session = None
        try:
            session = _make_session()
            df = yf.download(
                tickers=ticker,
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=end_exclusive.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                session=session,
            )
            return _normalize(df)
        except Exception as exc:  # yfinance is loose about its exception types
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < max_retries - 1:
                delay = min(BACKOFF_BASE_S * (2 ** attempt), BACKOFF_CAP_S)
                jitter = random.uniform(0, 0.5)
                logger.warning(
                    "yfinance 429 on %s (attempt %d/%d). Sleeping %.1fs before retry.",
                    ticker, attempt + 1, max_retries, delay + jitter,
                )
                time.sleep(delay + jitter)
                continue
            # Non-429 error or final attempt: give up.
            break
        finally:
            # curl_cffi sessions hold a libcurl handle; close them eagerly.
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

    if last_exc is not None and _is_rate_limit_error(last_exc):
        raise YFinanceRateLimited(
            f"yfinance rate-limited on {ticker} after {max_retries} retries"
        ) from last_exc
    raise YFinanceFetchError(f"yfinance fetch failed for {ticker}: {last_exc}") from last_exc
