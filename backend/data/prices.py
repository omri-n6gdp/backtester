"""
Public price-fetching interface.

`get_prices(ticker, timeframe, start, end)` returns an OHLCV DataFrame.

Strategy:
1. Read existing Parquet cache for (ticker, timeframe).
2. Compute the gap between the cached tail and `end`.
3. Fetch only the gap from yfinance (with throttle/backoff).
4. On yfinance rate-limit failure, fall back to Stooq for daily data.
5. Append, deduplicate, write back atomically.
6. Slice to [start, end] and return.

Cache layout: ~/.backtester/prices/{timeframe}/{ticker}.parquet
Metadata:    ~/.backtester/prices/meta.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from . import stooq_source, yfinance_source
from .yfinance_source import (
    YFinanceFetchError,
    YFinanceRateLimited,
    _empty_frame,
)

logger = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".backtester" / "prices"
META_PATH = CACHE_ROOT / "meta.json"
SUPPORTED_TIMEFRAMES = ("1d", "1h", "5m")


# --- Cache I/O -------------------------------------------------------------

def _cache_path(ticker: str, timeframe: str) -> Path:
    return CACHE_ROOT / timeframe / f"{ticker}.parquet"


def _read_cache(ticker: str, timeframe: str) -> pd.DataFrame:
    path = _cache_path(ticker, timeframe)
    if not path.exists():
        return _empty_frame()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Cache read failed for %s/%s: %s. Treating as empty.",
                       timeframe, ticker, exc)
        return _empty_frame()

    # Defensive normalization in case the cache predates a schema change.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"
    return df.sort_index()


def _write_cache_atomic(df: pd.DataFrame, ticker: str, timeframe: str) -> None:
    """Write Parquet via temp file + rename so concurrent readers never see a half-written file."""
    path = _cache_path(ticker, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{ticker}.", suffix=".parquet.tmp", dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=True)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _read_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except json.JSONDecodeError:
        logger.warning("meta.json is corrupt; rebuilding.")
        return {}


def _write_meta(meta: dict) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".meta.", suffix=".json.tmp", dir=str(META_PATH.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp_path, META_PATH)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _meta_key(ticker: str, timeframe: str) -> str:
    return f"{timeframe}:{ticker}"


def _update_meta(ticker: str, timeframe: str, last_ts: Optional[pd.Timestamp]) -> None:
    meta = _read_meta()
    meta[_meta_key(ticker, timeframe)] = {
        "last_timestamp": last_ts.isoformat() if last_ts is not None else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_meta(meta)


# --- Gap computation -------------------------------------------------------

def _need_fetch(
    cached: pd.DataFrame, start: date, end: date, timeframe: str,
) -> tuple[Optional[date], Optional[date]]:
    """
    Decide whether to fetch and over what window.

    Returns (fetch_start, fetch_end) or (None, None) if cache fully covers the request.
    """
    if cached.empty:
        return start, end

    cached_start = cached.index.min().date()
    cached_end = cached.index.max().date()

    # Fully covered.
    if cached_start <= start and cached_end >= end:
        return None, None

    # Need to extend forward (the common case).
    if cached_end < end:
        # Re-fetch from one bar before the cached tail to handle late corrections.
        # For daily this is one day; for intraday, one day's overlap is harmless.
        gap_start = cached_end - timedelta(days=1)
        return gap_start, end

    # Need to extend backwards (rarer; user asked for older data than we have).
    if cached_start > start:
        return start, cached_start

    return start, end  # shouldn't reach here, but be safe


# --- Public API ------------------------------------------------------------

def get_prices(
    ticker: str,
    timeframe: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Return OHLCV for [start, end] for the given (ticker, timeframe).

    Reads cache, fetches the gap, writes back, returns the slice.

    Args:
        ticker: e.g. 'SPY', 'BTC-USD'.
        timeframe: '1d', '1h', or '5m'.
        start: inclusive start date.
        end: inclusive end date.

    Raises:
        ValueError: bad timeframe or start > end.
        YFinanceFetchError / StooqFetchError: if all sources fail.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {SUPPORTED_TIMEFRAMES}")
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    cached = _read_cache(ticker, timeframe)
    fetch_start, fetch_end = _need_fetch(cached, start, end, timeframe)

    if fetch_start is None:
        logger.debug("Cache hit for %s/%s [%s, %s]", ticker, timeframe, start, end)
        return _slice(cached, start, end)

    logger.info("Fetching %s/%s gap [%s, %s]", ticker, timeframe, fetch_start, fetch_end)

    new_data: Optional[pd.DataFrame] = None
    try:
        new_data = yfinance_source.fetch(ticker, timeframe, fetch_start, fetch_end)
    except YFinanceRateLimited as exc:
        if timeframe == "1d":
            logger.warning("yfinance rate-limited; falling back to Stooq for %s.", ticker)
            try:
                new_data = stooq_source.fetch_daily(ticker, fetch_start, fetch_end)
            except stooq_source.StooqFetchError as stooq_exc:
                # If we have any cached data, return it rather than crash.
                if not cached.empty:
                    logger.error("Both yfinance and Stooq failed for %s; returning stale cache.", ticker)
                    return _slice(cached, start, end)
                raise stooq_exc from exc
        else:
            # No intraday fallback. If we have cache, return it; else propagate.
            if not cached.empty:
                logger.error("yfinance rate-limited and no intraday fallback for %s; returning stale cache.", ticker)
                return _slice(cached, start, end)
            raise

    if new_data is None or new_data.empty:
        # Nothing new — record the fetch attempt and return cache.
        _update_meta(ticker, timeframe, cached.index.max() if not cached.empty else None)
        return _slice(cached, start, end)

    # Merge: dedupe on timestamp index, prefer the new data.
    merged = pd.concat([cached, new_data])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    _write_cache_atomic(merged, ticker, timeframe)
    _update_meta(ticker, timeframe, merged.index.max())

    return _slice(merged, start, end)


def _slice(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df.empty:
        return df
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return df.loc[start_ts:end_ts].copy()
