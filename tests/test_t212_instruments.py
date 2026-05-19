"""Unit tests for backend.t212.instruments cache layer.

Tests redirect CACHE_ROOT to tmp_path and monkeypatch t212_client.get_instruments
to a deterministic in-memory list — no real T212 calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.t212 import client as t212_client
from backend.t212 import instruments


SAMPLE_INSTRUMENTS = [
    {
        "ticker": "AAPL_US_EQ", "shortName": "AAPL", "type": "STOCK",
        "name": "Apple Inc.", "currencyCode": "USD",
        "isin": "US0378331005", "workingScheduleId": 56,
        "maxOpenQuantity": 1000.0, "extendedHours": True,
        "addedOn": "2018-01-01T00:00:00.000+00:00",
    },
    {
        "ticker": "AAPLd_EQ", "shortName": "AAPL", "type": "STOCK",
        "name": "Apple Inc. (Deutsche)", "currencyCode": "EUR",
        "isin": "US0378331005", "workingScheduleId": 172,
        "maxOpenQuantity": 500.0, "extendedHours": False,
        "addedOn": "2020-06-01T00:00:00.000+00:00",
    },
    {
        "ticker": "SPY_US_EQ", "shortName": "SPY", "type": "ETF",
        "name": "SPDR S&P 500 ETF", "currencyCode": "USD",
        "isin": "US78462F1030", "workingScheduleId": 56,
        "maxOpenQuantity": 10000.0, "extendedHours": True,
        "addedOn": "2018-01-01T00:00:00.000+00:00",
    },
]


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the instruments cache to tmp_path. Yields the tmp_path."""
    cache_root = tmp_path / "t212_cache"
    cache_path = cache_root / "instruments.parquet"
    meta_path = cache_root / "meta.json"

    monkeypatch.setattr(instruments, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(instruments, "CACHE_PATH", cache_path)
    monkeypatch.setattr(instruments, "META_PATH", meta_path)

    yield cache_root


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """Stub out t212_client.get_instruments. Yields a controller:
        controller.set_return(list_or_callable)
        controller.set_error(exception)
        controller.call_count
    """
    class Controller:
        def __init__(self) -> None:
            self.call_count = 0
            self._return: object = SAMPLE_INSTRUMENTS
            self._error: BaseException | None = None

        def set_return(self, payload) -> None:
            self._return = payload
            self._error = None

        def set_error(self, exc: BaseException) -> None:
            self._error = exc

        def __call__(self) -> list[dict]:
            self.call_count += 1
            if self._error is not None:
                raise self._error
            return self._return  # type: ignore[return-value]

    ctrl = Controller()
    monkeypatch.setattr(t212_client, "get_instruments", ctrl)
    yield ctrl


# --- Freshness check ------------------------------------------------------


def test_cache_fresh_false_when_no_cache_file(tmp_cache):
    assert instruments._cache_fresh() is False


def test_cache_fresh_false_when_meta_missing(tmp_cache):
    # Touch the parquet file without writing meta — should NOT count as fresh.
    instruments.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    instruments.CACHE_PATH.touch()
    assert instruments._cache_fresh() is False


def test_cache_fresh_true_under_24h(tmp_cache, fake_client):
    instruments.get_instruments()  # populates cache
    assert instruments._cache_fresh() is True


def test_cache_fresh_false_over_24h(tmp_cache, fake_client):
    instruments.get_instruments()
    # Backdate the meta to make it stale.
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    instruments.META_PATH.write_text(json.dumps({"fetched_at": old}))
    assert instruments._cache_fresh() is False


# --- get_instruments() paths ---------------------------------------------


def test_first_call_hits_client_and_writes_cache(tmp_cache, fake_client):
    result = instruments.get_instruments()
    assert result == SAMPLE_INSTRUMENTS
    assert fake_client.call_count == 1
    assert instruments.CACHE_PATH.exists()
    assert instruments.META_PATH.exists()


def test_second_call_uses_cache(tmp_cache, fake_client):
    instruments.get_instruments()  # populates
    fake_client.call_count = 0     # reset
    result = instruments.get_instruments()
    assert result == SAMPLE_INSTRUMENTS
    assert fake_client.call_count == 0, "fresh cache should not hit the client"


def test_force_refresh_bypasses_cache(tmp_cache, fake_client):
    instruments.get_instruments()
    fake_client.call_count = 0
    instruments.get_instruments(force_refresh=True)
    assert fake_client.call_count == 1


def test_refresh_failure_falls_back_to_stale_cache(tmp_cache, fake_client):
    """Stale cache wins over an outage — the Live view stays usable."""
    instruments.get_instruments()  # populate cache
    # Backdate to force a refresh.
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    instruments.META_PATH.write_text(json.dumps({"fetched_at": old}))
    # Now make the refresh fail.
    fake_client.set_error(t212_client.T212UpstreamError("T212 is down"))

    result = instruments.get_instruments()
    assert result == SAMPLE_INSTRUMENTS, "should serve stale cache"


def test_refresh_failure_with_no_cache_propagates(tmp_cache, fake_client):
    fake_client.set_error(t212_client.T212RateLimited("rate limited"))
    with pytest.raises(t212_client.T212RateLimited):
        instruments.get_instruments()


def test_cache_round_trips_full_schema(tmp_cache, fake_client):
    """Every field in the source list survives the parquet round-trip
    intact — booleans, floats, ISO-string dates, etc."""
    instruments.get_instruments()
    fake_client.call_count = 0
    second = instruments.get_instruments()
    # Compare by primary key + a couple of representative fields.
    pairs = {(e["ticker"], e["currencyCode"], e["extendedHours"]) for e in second}
    expected = {(e["ticker"], e["currencyCode"], e["extendedHours"]) for e in SAMPLE_INSTRUMENTS}
    assert pairs == expected


# --- find_by_ticker -------------------------------------------------------


def test_find_by_ticker_exact_short_name_case_insensitive(tmp_cache, fake_client):
    # 'AAPL' matches both AAPL_US_EQ and AAPLd_EQ via shortName.
    matches = instruments.find_by_ticker("aapl")
    assert {m["ticker"] for m in matches} == {"AAPL_US_EQ", "AAPLd_EQ"}


def test_find_by_ticker_exact_t212_code(tmp_cache, fake_client):
    matches = instruments.find_by_ticker("AAPL_US_EQ")
    assert len(matches) == 1
    assert matches[0]["currencyCode"] == "USD"


def test_find_by_ticker_no_match(tmp_cache, fake_client):
    assert instruments.find_by_ticker("DOESNOTEXIST") == []


def test_find_by_ticker_empty_query(tmp_cache, fake_client):
    """Blank query short-circuits — doesn't accidentally match shortName=''."""
    assert instruments.find_by_ticker("") == []
    assert instruments.find_by_ticker("   ") == []


def test_find_by_ticker_no_substring_match(tmp_cache, fake_client):
    """'AAP' should not match 'AAPL' — exact match only by design."""
    assert instruments.find_by_ticker("AAP") == []


def test_find_by_ticker_with_explicit_list_skips_cache(tmp_cache, fake_client):
    """Pre-fetching a list and reusing it across calls hits the client once."""
    inst = instruments.get_instruments()
    fake_client.call_count = 0
    instruments.find_by_ticker("SPY", instruments=inst)
    instruments.find_by_ticker("AAPL", instruments=inst)
    assert fake_client.call_count == 0
