"""Unit tests for the data layer (prices.py + yfinance_source + stooq_source)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backend.data import prices as prices_mod
from backend.data import stooq_source, yfinance_source
from backend.data.yfinance_source import YFinanceRateLimited


# --- Helpers ---------------------------------------------------------------

def _mk_df(dates: list[str], base: float = 100.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates, utc=True), name="timestamp")
    n = len(idx)
    return pd.DataFrame(
        {
            "open":   [base + i for i in range(n)],
            "high":   [base + i + 1 for i in range(n)],
            "low":    [base + i - 1 for i in range(n)],
            "close":  [base + i + 0.5 for i in range(n)],
            "volume": [1_000_000.0 + i for i in range(n)],
        },
        index=idx,
    )


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redirect the cache root to a temp dir for hermetic tests."""
    fake_root = tmp_path / "prices"
    fake_root.mkdir()
    for tf in ("1d", "1h", "5m"):
        (fake_root / tf).mkdir()
    monkeypatch.setattr(prices_mod, "CACHE_ROOT", fake_root)
    monkeypatch.setattr(prices_mod, "META_PATH", fake_root / "meta.json")
    yield fake_root


# --- Tests -----------------------------------------------------------------

def test_cache_miss_writes_parquet(tmp_cache):
    """First fetch: cache empty -> yfinance called -> Parquet written."""
    fetched = _mk_df(["2024-01-02", "2024-01-03", "2024-01-04"])
    with patch.object(yfinance_source, "fetch", return_value=fetched) as mock_yf:
        out = prices_mod.get_prices("FAKE", "1d", date(2024, 1, 2), date(2024, 1, 4))

    mock_yf.assert_called_once()
    assert len(out) == 3
    assert (tmp_cache / "1d" / "FAKE.parquet").exists()


def test_cache_hit_skips_fetch(tmp_cache):
    """Second fetch with the cache fully covering the range: yfinance NOT called."""
    cached = _mk_df(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    cached.to_parquet(tmp_cache / "1d" / "FAKE.parquet")

    with patch.object(yfinance_source, "fetch") as mock_yf:
        out = prices_mod.get_prices("FAKE", "1d", date(2024, 1, 3), date(2024, 1, 4))

    mock_yf.assert_not_called()
    assert len(out) == 2
    assert out.index.min().date() == date(2024, 1, 3)
    assert out.index.max().date() == date(2024, 1, 4)


def test_partial_cache_fetches_gap(tmp_cache):
    """Cache has Jan 2-3; user asks Jan 2-5 -> only Jan 2-5 gap (with overlap) fetched."""
    cached = _mk_df(["2024-01-02", "2024-01-03"])
    cached.to_parquet(tmp_cache / "1d" / "FAKE.parquet")

    new = _mk_df(["2024-01-03", "2024-01-04", "2024-01-05"], base=200.0)
    with patch.object(yfinance_source, "fetch", return_value=new) as mock_yf:
        out = prices_mod.get_prices("FAKE", "1d", date(2024, 1, 2), date(2024, 1, 5))

    assert mock_yf.call_count == 1
    assert len(out) == 4  # Jan 2, 3, 4, 5
    # Dedup: the new data for Jan 3 should win (base=200, close = 200.5).
    assert out.loc[pd.Timestamp("2024-01-03", tz="UTC"), "close"] == 200.5


def test_dedup_keeps_latest(tmp_cache):
    """Overlapping fetches must dedup, preferring the freshly-fetched row."""
    cached = _mk_df(["2024-01-02", "2024-01-03"], base=100.0)
    cached.to_parquet(tmp_cache / "1d" / "FAKE.parquet")

    overlapping = _mk_df(["2024-01-02", "2024-01-03"], base=999.0)
    with patch.object(yfinance_source, "fetch", return_value=overlapping):
        out = prices_mod.get_prices("FAKE", "1d", date(2024, 1, 1), date(2024, 1, 5))

    # Cache extension forces a fetch (Jan 5 is past cached tail).
    # After dedup, the overlapping rows should win for Jan 2-3.
    assert out.loc[pd.Timestamp("2024-01-02", tz="UTC"), "close"] == 999.5


def test_stooq_fallback_on_yfinance_rate_limit(tmp_cache):
    """Daily fetch: yfinance 429 -> Stooq is used."""
    stooq_data = _mk_df(["2024-01-02", "2024-01-03"], base=500.0)
    with patch.object(yfinance_source, "fetch", side_effect=YFinanceRateLimited("429")), \
         patch.object(stooq_source, "fetch_daily", return_value=stooq_data) as mock_stooq:
        out = prices_mod.get_prices("FAKE", "1d", date(2024, 1, 2), date(2024, 1, 3))

    mock_stooq.assert_called_once()
    assert len(out) == 2
    assert out.iloc[0]["close"] == 500.5


def test_no_stooq_fallback_for_intraday(tmp_cache):
    """Intraday: yfinance 429 with no cache -> raise, since Stooq is daily-only."""
    with patch.object(yfinance_source, "fetch", side_effect=YFinanceRateLimited("429")):
        with pytest.raises(YFinanceRateLimited):
            prices_mod.get_prices("FAKE", "1h", date(2024, 1, 2), date(2024, 1, 3))


def test_invalid_timeframe_raises(tmp_cache):
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        prices_mod.get_prices("SPY", "30m", date(2024, 1, 1), date(2024, 1, 2))


def test_start_after_end_raises(tmp_cache):
    with pytest.raises(ValueError, match="start"):
        prices_mod.get_prices("SPY", "1d", date(2024, 1, 5), date(2024, 1, 1))


def test_normalize_drops_all_nan_rows():
    """yfinance occasionally returns rows of all-NaN; the normalizer must drop them."""
    raw = pd.DataFrame(
        {
            "Open":   [100.0, None, 102.0],
            "High":   [101.0, None, 103.0],
            "Low":    [99.0,  None, 101.0],
            "Close":  [100.5, None, 102.5],
            "Volume": [1e6,   None, 1e6],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    out = yfinance_source._normalize(raw)
    assert len(out) == 2
    assert out.index.tz is not None
    assert out.index.name == "timestamp"
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_normalize_handles_naive_index():
    """Naive (tz-unaware) indexes must be localized to UTC."""
    raw = pd.DataFrame(
        {
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [1e6],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    out = yfinance_source._normalize(raw)
    assert str(out.index.tz) == "UTC"


def test_meta_updated_after_fetch(tmp_cache):
    """meta.json should record the last_timestamp and fetched_at after a successful fetch."""
    fetched = _mk_df(["2024-01-02", "2024-01-03"])
    with patch.object(yfinance_source, "fetch", return_value=fetched):
        prices_mod.get_prices("FAKE", "1d", date(2024, 1, 2), date(2024, 1, 3))

    import json
    meta = json.loads((tmp_cache / "meta.json").read_text())
    assert "1d:FAKE" in meta
    assert meta["1d:FAKE"]["last_timestamp"] is not None
    assert meta["1d:FAKE"]["fetched_at"] is not None
