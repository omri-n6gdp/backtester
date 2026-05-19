"""
Daily Parquet cache for the T212 instrument metadata list.

The full list is ~17K entries / several MB; T212 only adds or removes
instruments on a multi-day cadence, so a 24h refresh cycle is plenty.
Pattern mirrors backend.data.prices: read cache → check freshness →
refetch on miss → atomic write back. Stale cache wins over a failed
refresh, so a transient T212 outage doesn't hide the Live view.

Cache layout:
    ~/.backtester/t212/instruments.parquet
    ~/.backtester/t212/meta.json   ({"fetched_at": ISO8601 UTC})

Schema (each entry): ticker, shortName, name, type, currencyCode, isin,
workingScheduleId, maxOpenQuantity, extendedHours, addedOn.
All ten fields are 100%-populated on the live API as of 2026-05.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.t212 import client as t212_client

CACHE_ROOT = Path.home() / ".backtester" / "t212"
CACHE_PATH = CACHE_ROOT / "instruments.parquet"
META_PATH = CACHE_ROOT / "meta.json"
MAX_AGE = timedelta(hours=24)

logger = logging.getLogger(__name__)


# --- Cache I/O ------------------------------------------------------------


def _read_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except json.JSONDecodeError:
        logger.warning("T212 instruments meta.json is corrupt; treating as missing.")
        return {}


def _write_meta(meta: dict[str, Any]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".meta.", suffix=".json.tmp", dir=str(META_PATH.parent),
    )
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


def _cache_fresh() -> bool:
    """True iff a cache file exists and was written < MAX_AGE ago."""
    if not CACHE_PATH.exists():
        return False
    fetched_at = _read_meta().get("fetched_at")
    if not fetched_at:
        return False
    try:
        fetched_dt = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if fetched_dt.tzinfo is None:
        fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched_dt < MAX_AGE


def _read_cache() -> list[dict[str, Any]]:
    df = pd.read_parquet(CACHE_PATH)
    return df.to_dict("records")


def _write_cache(instruments: list[dict[str, Any]]) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(instruments)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".instruments.", suffix=".parquet.tmp", dir=str(CACHE_ROOT),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, CACHE_PATH)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# --- Public API -----------------------------------------------------------


def get_instruments(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Return the full T212 instrument list.

    Strategy:
      1. If cache is fresh (< 24h old) and not forced, serve from disk.
      2. Otherwise hit T212, persist, and serve the fresh list.
      3. If the refresh fetch fails AND a (stale) cache exists, return
         the stale list rather than failing the caller. Surface the
         error in logs.
      4. If the refresh fails AND no cache exists, propagate the error.

    Args:
        force_refresh: bypass the freshness check; always refetch.

    Raises:
        T212Error subclasses from backend.t212.client, only when no
        cached fallback is available.
    """
    if not force_refresh and _cache_fresh():
        logger.debug("T212 instruments cache hit")
        return _read_cache()

    logger.info("T212 instruments refresh (force=%s)", force_refresh)
    try:
        fresh = t212_client.get_instruments()
    except t212_client.T212Error:
        if CACHE_PATH.exists():
            logger.warning(
                "T212 instruments refresh failed; returning stale cache.",
                exc_info=True,
            )
            return _read_cache()
        raise

    _write_cache(fresh)
    _write_meta({"fetched_at": datetime.now(timezone.utc).isoformat()})
    return fresh


def find_by_ticker(
    query: str,
    *,
    instruments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Look up instruments by human ticker (`shortName`) OR T212 code (`ticker`).

    Match is case-insensitive and exact (no substring match — that would
    drown the caller in noise on common short names like 'A' or 'B').
    Returns a list because a single user-friendly ticker can map to
    multiple T212 venues (e.g. 'AAPL' on US and German exchanges).

    Args:
        query: user-supplied ticker. Whitespace is stripped.
        instruments: optional pre-fetched list. Pass to avoid repeated
            cache reads when looking up many tickers in a loop.

    Returns:
        Zero, one, or many matching instrument dicts. Order preserved
        from the source list.
    """
    if not query or not query.strip():
        return []
    needle = query.strip().casefold()
    source = instruments if instruments is not None else get_instruments()

    matches: list[dict[str, Any]] = []
    for entry in source:
        ticker = str(entry.get("ticker") or "").casefold()
        short = str(entry.get("shortName") or "").casefold()
        if needle == ticker or needle == short:
            matches.append(entry)
    return matches
