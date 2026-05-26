"""
Phase 18.2 — read / inspect / evict / compact the local price cache.

A thin wrapper over backend.data.prices' on-disk layout so the API can
expose:

  - list_cache_entries() — current contents of ~/.backtester/prices/
                           with size, row count, and last-fetched ts.
  - evict_cache_entry()  — atomically remove one parquet + its meta row.
  - compact_cache()      — re-write every parquet via the data layer's
                           own helpers (sort + dedupe + recompression).

Used by the routes under /cache/* (Phase 18.2). The maintenance script
`scripts/compact_cache.py` keeps working — it's just a CLI shim that
ends up exercising the same logic.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backend.data.prices import (
    CACHE_ROOT,
    _meta_key,
    _read_cache,
    _read_meta,
    _write_cache_atomic,
    _write_meta,
    _cache_path,
)

logger = logging.getLogger(__name__)


def list_cache_entries() -> list[dict]:
    """Walk ~/.backtester/prices/{timeframe}/ and return one entry per
    cached Parquet. `n_rows` and `fetched_at` are best-effort — a
    corrupt Parquet leaves them null but the entry still appears."""
    if not CACHE_ROOT.exists():
        return []
    meta = _read_meta()
    out: list[dict] = []
    for tf_dir in sorted(CACHE_ROOT.iterdir()):
        if not tf_dir.is_dir():
            continue
        timeframe = tf_dir.name
        for f in sorted(tf_dir.glob("*.parquet")):
            ticker = f.stem
            entry = {
                "timeframe": timeframe,
                "ticker": ticker,
                "size_bytes": f.stat().st_size,
                "n_rows": None,
                "last_timestamp": None,
                "fetched_at": None,
            }
            mrow = meta.get(_meta_key(ticker, timeframe))
            if mrow:
                entry["last_timestamp"] = mrow.get("last_timestamp")
                entry["fetched_at"] = mrow.get("fetched_at")
            try:
                df = _read_cache(ticker, timeframe)
                entry["n_rows"] = int(len(df))
            except Exception:  # noqa: BLE001 — best-effort
                logger.exception("could not read row count for %s/%s", timeframe, ticker)
            out.append(entry)
    return out


def evict_cache_entry(timeframe: str, ticker: str) -> bool:
    """Remove one cached Parquet and clear its meta row. Returns True
    when a file actually existed, False otherwise (so the route can
    404 cleanly)."""
    path = _cache_path(ticker, timeframe)
    existed = path.exists()
    if existed:
        try:
            path.unlink()
        except OSError:
            logger.exception("failed to remove %s", path)
            raise

    meta = _read_meta()
    key = _meta_key(ticker, timeframe)
    if key in meta:
        del meta[key]
        _write_meta(meta)

    return existed


def compact_cache() -> dict:
    """Re-write every cached Parquet via _write_cache_atomic so any
    fragmentation is healed. Returns a summary {files, bytes_before,
    bytes_after}. Idempotent."""
    if not CACHE_ROOT.exists():
        return {"files": 0, "bytes_before": 0, "bytes_after": 0}

    bytes_before = 0
    bytes_after = 0
    count = 0
    for tf_dir in sorted(CACHE_ROOT.iterdir()):
        if not tf_dir.is_dir():
            continue
        for f in sorted(tf_dir.glob("*.parquet")):
            timeframe = tf_dir.name
            ticker = f.stem
            try:
                bytes_before += f.stat().st_size
                df = _read_cache(ticker, timeframe)
                _write_cache_atomic(df, ticker, timeframe)
                bytes_after += f.stat().st_size
                count += 1
            except Exception:  # noqa: BLE001
                logger.exception("compact: %s/%s failed", timeframe, ticker)

    return {"files": count, "bytes_before": bytes_before, "bytes_after": bytes_after}
