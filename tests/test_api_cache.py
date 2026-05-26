"""
Phase 18.2 — API tests for /cache/* (read / evict / compact).

Writes synthetic parquets into a tmp-redirected CACHE_ROOT so the real
~/.backtester/ tree is never touched.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backend.data import prices as prices_mod


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Redirect CACHE_ROOT (and the helpers that read it) to tmp."""
    root = tmp_path / "prices"
    root.mkdir()
    monkeypatch.setattr(prices_mod, "CACHE_ROOT", root)
    monkeypatch.setattr(prices_mod, "META_PATH", root / "meta.json")
    # cache_admin also imports CACHE_ROOT at module load.
    from backend.data import cache_admin
    monkeypatch.setattr(cache_admin, "CACHE_ROOT", root)
    return root


def _seed_cache(cache_root, timeframe: str, ticker: str, n: int = 100):
    """Write a small synthetic parquet via the real data layer helpers
    so meta.json + the file format match what the engine writes."""
    (cache_root / timeframe).mkdir(exist_ok=True)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1_000_000.0,
    }, index=idx).rename_axis("timestamp")
    prices_mod._write_cache_atomic(df, ticker, timeframe)
    prices_mod._update_meta(ticker, timeframe, df.index[-1])
    return df


def test_list_cache_returns_seeded_entries(cache_root, api_client):
    _seed_cache(cache_root, "1d", "SPY", n=100)
    _seed_cache(cache_root, "1d", "QQQ", n=50)
    _seed_cache(cache_root, "1h", "BTC-USD", n=30)

    body = api_client.get("/cache/prices").json()
    by_pair = {(e["timeframe"], e["ticker"]): e for e in body["entries"]}
    assert ("1d", "SPY") in by_pair
    assert ("1d", "QQQ") in by_pair
    assert ("1h", "BTC-USD") in by_pair
    spy = by_pair[("1d", "SPY")]
    assert spy["n_rows"] == 100
    assert spy["fetched_at"] is not None
    assert spy["size_bytes"] > 0


def test_evict_removes_file_and_meta_row(cache_root, api_client):
    _seed_cache(cache_root, "1d", "SPY")
    _seed_cache(cache_root, "1d", "QQQ")

    resp = api_client.request(
        "DELETE", "/cache/prices",
        json={"timeframe": "1d", "ticker": "SPY"},
    )
    assert resp.status_code == 200, resp.text

    body = api_client.get("/cache/prices").json()
    tickers = {(e["timeframe"], e["ticker"]) for e in body["entries"]}
    assert ("1d", "SPY") not in tickers
    assert ("1d", "QQQ") in tickers
    # File is gone.
    assert not (cache_root / "1d" / "SPY.parquet").exists()


def test_evict_unknown_entry_returns_404(cache_root, api_client):
    resp = api_client.request(
        "DELETE", "/cache/prices",
        json={"timeframe": "1d", "ticker": "GHOST"},
    )
    assert resp.status_code == 404


def test_compact_is_idempotent(cache_root, api_client):
    _seed_cache(cache_root, "1d", "SPY")
    _seed_cache(cache_root, "1d", "QQQ")

    body = api_client.post("/cache/compact").json()
    assert body["files"] == 2
    # Re-running compact a second time still works and reports the same
    # file count (no error, no growth).
    again = api_client.post("/cache/compact").json()
    assert again["files"] == 2
