"""
Stooq fallback fetcher (daily only).

Used when yfinance is rate-limited. Stooq provides free CSV endpoints with
no auth and lenient throttling. Only daily data is supported here.

Stooq ticker conventions differ from Yahoo; we apply a small mapping.
"""

from __future__ import annotations

import io
import logging
from datetime import date

import httpx
import pandas as pd

from .yfinance_source import _empty_frame  # reuse the canonical empty frame

logger = logging.getLogger(__name__)

STOOQ_URL = "https://stooq.com/q/d/l/"


class StooqFetchError(Exception):
    """Raised when Stooq returns nothing usable."""


def _to_stooq_symbol(ticker: str) -> str:
    """
    Map a Yahoo-style ticker to a Stooq symbol.

    - US equities/ETFs: AAPL -> aapl.us, SPY -> spy.us
    - Indices: ^GSPC -> ^spx (we don't bother — fall back manually if needed)
    - Crypto: BTC-USD -> btcusd
    - FX: EURUSD=X -> eurusd

    This map is intentionally conservative; unknown tickers pass through lowercased.
    """
    t = ticker.strip()
    if t.endswith("-USD") and "-" in t:
        # Crypto: BTC-USD -> btcusd
        return t.replace("-", "").lower()
    if t.endswith("=X"):
        # FX: EURUSD=X -> eurusd
        return t[:-2].lower()
    if t.startswith("^"):
        # Index: leave as-is, lowercased. Stooq has e.g. ^spx for S&P 500.
        return t.lower()
    # Default: assume US equity/ETF.
    return f"{t.lower()}.us"


def fetch_daily(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Stooq.

    Returns a normalized DataFrame matching the yfinance source schema.
    Raises StooqFetchError if nothing usable comes back.
    """
    symbol = _to_stooq_symbol(ticker)
    params = {
        "s": symbol,
        "i": "d",
        "d1": start.strftime("%Y%m%d"),
        "d2": end.strftime("%Y%m%d"),
    }

    logger.info("Stooq fallback: fetching %s (mapped to %s)", ticker, symbol)
    try:
        resp = httpx.get(STOOQ_URL, params=params, timeout=15.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise StooqFetchError(f"Stooq HTTP error for {ticker}: {exc}") from exc

    text = resp.text.strip()
    # Stooq returns "No data" as a plain text body when the symbol is unknown.
    if not text or text.lower().startswith("no data"):
        raise StooqFetchError(f"Stooq returned no data for {ticker} (symbol={symbol})")

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise StooqFetchError(f"Stooq CSV parse failed for {ticker}: {exc}") from exc

    if df.empty:
        return _empty_frame()

    # Stooq columns: Date, Open, High, Low, Close, Volume.
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise StooqFetchError(f"Stooq CSV missing 'date' column for {ticker}")

    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("timestamp").drop(columns=["date"])

    keep = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise StooqFetchError(f"Stooq CSV missing columns {missing} for {ticker}")
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    df = df.dropna(how="all").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
