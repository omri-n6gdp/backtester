"""
API tests for GET /library and GET /runs/{run_id} (Phase 8).
"""
from __future__ import annotations

import pytest

from tests.conftest import install_fake_prices, make_synthetic_ohlcv, post_strategy


SAMPLE_STRATEGY = """
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 5))
    long = int(params.get('long', 20))
    return (data['close'].rolling(short).mean() > data['close'].rolling(long).mean()).astype(float)
"""


@pytest.fixture
def library_client(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=500))
    sid = post_strategy(api_client, "SMA", SAMPLE_STRATEGY)
    return api_client, sid


# ---- /library ----------------------------------------------------------

def test_library_empty_when_no_runs(api_client):
    resp = api_client.get("/library")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_library_lists_runs_after_backtest(library_client):
    client, sid = library_client
    # Run two backtests on different tickers so breadth_score can populate.
    for ticker in ["SPY", "AAPL"]:
        resp = client.post("/backtest", json={
            "strategy_id": sid, "ticker": ticker, "timeframe": "1d",
            "start": "2010-01-01", "end": "2012-01-01",
            "params": {"short": 5, "long": 20},
        })
        assert resp.status_code == 200, resp.text

    body = client.get("/library").json()
    assert len(body["runs"]) == 2
    tickers = sorted(r["ticker"] for r in body["runs"])
    assert tickers == ["AAPL", "SPY"]
    # All scoring components present and in [0, 1] (or null for dsr).
    for r in body["runs"]:
        for key in ("sharpe_score", "calmar_score", "consistency_score",
                    "breadth_score", "recency_score"):
            v = r[key]
            assert 0.0 <= v <= 1.0, f"{key}={v} out of range"
        assert r["dsr_score"] is None
        # Strategy was tested on 2 distinct tickers; breadth is between 0 and 1.
        assert 0.0 < r["breadth_score"] < 1.0


def test_library_ticker_filter(library_client):
    client, sid = library_client
    for ticker in ["SPY", "AAPL", "QQQ"]:
        client.post("/backtest", json={
            "strategy_id": sid, "ticker": ticker, "timeframe": "1d",
            "start": "2010-01-01", "end": "2012-01-01",
            "params": {"short": 5, "long": 20},
        })

    body = client.get("/library", params={"ticker": "SPY"}).json()
    assert len(body["runs"]) == 1
    assert body["runs"][0]["ticker"] == "SPY"


def test_library_dedupes_identical_config(library_client):
    """Re-running the exact same config must not pile up library rows."""
    client, sid = library_client
    cfg = {
        "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2012-01-01",
        "params": {"short": 5, "long": 20},
    }
    first = client.post("/backtest", json=cfg).json()["run_id"]
    second = client.post("/backtest", json=cfg).json()["run_id"]
    assert first != second  # both persisted — history is kept in the DB

    runs = client.get("/library").json()["runs"]
    assert len(runs) == 1, "identical config should collapse to one row"
    # The surviving row is the most recent run, not the first.
    assert runs[0]["run_id"] == second


def test_library_keeps_distinct_param_configs(library_client):
    """Different params on the same ticker are distinct configs — not deduped."""
    client, sid = library_client
    for long in (20, 30):
        resp = client.post("/backtest", json={
            "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
            "start": "2010-01-01", "end": "2012-01-01",
            "params": {"short": 5, "long": long},
        })
        assert resp.status_code == 200, resp.text

    runs = client.get("/library").json()["runs"]
    assert len(runs) == 2


def test_library_carries_favourite_flag(library_client):
    """A run's row reflects whether its strategy is favourited."""
    client, sid = library_client
    resp = client.post("/backtest", json={
        "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2012-01-01",
        "params": {"short": 5, "long": 20},
    })
    assert resp.status_code == 200, resp.text

    assert client.get("/library").json()["runs"][0]["favourite"] is False
    client.post(f"/strategies/{sid}/favourite", json={"favourite": True})
    assert client.get("/library").json()["runs"][0]["favourite"] is True


def test_library_keeps_distinct_date_ranges(library_client):
    """Same params but a different window is a distinct config — not deduped."""
    client, sid = library_client
    for end in ("2012-01-01", "2013-01-01"):
        resp = client.post("/backtest", json={
            "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
            "start": "2010-01-01", "end": end,
            "params": {"short": 5, "long": 20},
        })
        assert resp.status_code == 200, resp.text

    runs = client.get("/library").json()["runs"]
    assert len(runs) == 2


# ---- /runs/{run_id} ----------------------------------------------------

def test_get_run_roundtrips_a_backtest(library_client):
    client, sid = library_client
    resp = client.post("/backtest", json={
        "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2012-01-01",
        "params": {"short": 5, "long": 20},
    })
    run_id = resp.json()["run_id"]
    original = resp.json()

    got = client.get(f"/runs/{run_id}").json()
    assert got["run_id"] == run_id
    assert len(got["timestamps"]) == len(original["timestamps"])
    assert got["equity"][-1] == original["equity"][-1]
    assert got["summary"]["sharpe"] == original["summary"]["sharpe"]
    assert len(got["trades"]) == len(original["trades"])
    # The runs route doesn't replay optional analyses — they're null.
    assert got["oos"] is None
    assert got["monte_carlo"] is None
    assert got["deflated_sharpe"] is None
    assert got["regimes"] is None


def test_get_run_unknown_id_returns_404(api_client):
    resp = api_client.get("/runs/nonexistent")
    assert resp.status_code == 404


# ---- Phase 16.3 — notes + tags ----------------------------------------


def _run_once(library_client, **overrides):
    client, sid = library_client
    payload = {
        "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
        "start": "2010-01-01", "end": "2012-01-01",
        "params": {"short": 5, "long": 20},
    }
    payload.update(overrides)
    return client.post("/backtest", json=payload).json()


def test_patch_run_sets_notes_and_tags(library_client):
    client, _ = library_client
    body = _run_once(library_client)
    run_id = body["run_id"]
    # Initially empty.
    assert body["notes"] is None
    assert body["tags"] == []

    upd = client.patch(
        f"/runs/{run_id}",
        json={"notes": "depends on covid rally", "tags": ["fragile", "needs-review"]},
    )
    assert upd.status_code == 200, upd.text
    assert upd.json()["notes"] == "depends on covid rally"
    assert upd.json()["tags"] == ["fragile", "needs-review"]

    # Round-trips via GET /runs/{id}.
    got = client.get(f"/runs/{run_id}").json()
    assert got["notes"] == "depends on covid rally"
    assert got["tags"] == ["fragile", "needs-review"]


def test_patch_run_normalizes_tags(library_client):
    client, _ = library_client
    body = _run_once(library_client)
    run_id = body["run_id"]
    upd = client.patch(
        f"/runs/{run_id}",
        json={"tags": ["  Fragile  ", "fragile", "Needs Review!"]},
    )
    # Lowercased, dedup'd, non-alnum stripped (space → empty).
    assert upd.json()["tags"] == ["fragile", "needsreview"]


def test_patch_run_omitted_field_left_unchanged(library_client):
    client, _ = library_client
    body = _run_once(library_client)
    run_id = body["run_id"]
    client.patch(f"/runs/{run_id}", json={"notes": "first", "tags": ["x"]})
    # Update only notes; tags stay.
    upd = client.patch(f"/runs/{run_id}", json={"notes": "second"})
    assert upd.json()["notes"] == "second"
    assert upd.json()["tags"] == ["x"]
    # Update only tags; notes stay.
    upd2 = client.patch(f"/runs/{run_id}", json={"tags": ["y"]})
    assert upd2.json()["notes"] == "second"
    assert upd2.json()["tags"] == ["y"]


def test_patch_run_explicit_null_clears(library_client):
    client, _ = library_client
    body = _run_once(library_client)
    run_id = body["run_id"]
    client.patch(f"/runs/{run_id}", json={"notes": "x", "tags": ["a"]})
    cleared = client.patch(f"/runs/{run_id}", json={"notes": None, "tags": None})
    assert cleared.json()["notes"] is None
    assert cleared.json()["tags"] == []


def test_library_returns_notes_and_tags(library_client):
    client, _ = library_client
    body = _run_once(library_client)
    run_id = body["run_id"]
    client.patch(
        f"/runs/{run_id}",
        json={"notes": "fragile", "tags": ["fragile"]},
    )
    rows = client.get("/library").json()["runs"]
    target = next(r for r in rows if r["run_id"] == run_id)
    assert target["notes"] == "fragile"
    assert target["tags"] == ["fragile"]


def test_patch_run_unknown_id_returns_404(library_client):
    client, _ = library_client
    resp = client.patch("/runs/nonexistent", json={"notes": "x"})
    assert resp.status_code == 404
