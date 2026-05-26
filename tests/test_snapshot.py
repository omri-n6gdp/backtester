"""
Phase 17.3 — unit + API tests for as-of-date snapshots.
"""
from __future__ import annotations

import pandas as pd

from backend.engine.snapshot import compute_yearly_snapshots
from tests.conftest import install_fake_prices, make_synthetic_ohlcv, post_strategy


# ---- engine unit tests --------------------------------------------------


def _equity_series(start: str, n: int, growth_per_bar: float = 0.0005):
    """A monotonically growing equity curve from `start`, `n` business days."""
    idx = pd.date_range(start=start, periods=n, freq="B", tz="UTC")
    eq = (1.0 + growth_per_bar) ** pd.Series(range(n)) * 100_000.0
    eq.index = idx
    return idx, eq


def test_snapshot_per_calendar_year():
    """A 3-year run produces 3 snapshots, latest-year last."""
    n = 252 * 3
    idx, eq = _equity_series("2020-01-01", n)
    returns = eq.pct_change().fillna(0.0)
    snaps = compute_yearly_snapshots(
        [t.isoformat() for t in idx],
        [float(v) for v in eq.to_list()],
        [float(v) for v in returns.to_list()],
        bars_per_year=252,
    )
    years = [s.year for s in snaps]
    assert years == sorted(years)
    assert len(snaps) == 3
    # CAGR-to-date is positive on a monotone curve.
    assert all(s.cagr_to_date > 0 for s in snaps)


def test_snapshot_cumulative_metrics_grow_through_time():
    """For a steady positive-return curve, total return-to-date should
    grow year over year."""
    idx, eq = _equity_series("2020-01-01", 252 * 4)
    returns = eq.pct_change().fillna(0.0)
    snaps = compute_yearly_snapshots(
        [t.isoformat() for t in idx],
        [float(v) for v in eq.to_list()],
        [float(v) for v in returns.to_list()],
    )
    rets = [s.total_return_to_date for s in snaps]
    assert rets == sorted(rets)


def test_snapshot_empty_input_returns_empty():
    assert compute_yearly_snapshots([], [], []) == []


# ---- API route ----------------------------------------------------------


SMA_STRATEGY = """
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 5))
    long = int(params.get('long', 20))
    return (data['close'].rolling(short).mean() > data['close'].rolling(long).mean()).astype(float)
"""


def test_run_snapshots_route_returns_one_per_year(api_client, monkeypatch):
    """A 3-year backtest → /runs/{id}/snapshots returns 3 entries."""
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=252 * 3))
    sid = post_strategy(api_client, "SMA", SMA_STRATEGY)
    run_id = api_client.post("/backtest", json={
        "strategy_id": sid, "ticker": "SPY", "timeframe": "1d",
        "start": "2020-01-01", "end": "2023-01-01",
        "params": {"short": 5, "long": 20},
    }).json()["run_id"]

    body = api_client.get(f"/runs/{run_id}/snapshots").json()
    assert body["run_id"] == run_id
    assert len(body["snapshots"]) == 3
    for s in body["snapshots"]:
        assert {"year", "year_end", "n_bars", "cagr_to_date",
                "sharpe_to_date", "max_drawdown_to_date"} <= set(s.keys())


def test_run_snapshots_unknown_id_returns_404(api_client):
    resp = api_client.get("/runs/nonexistent/snapshots")
    assert resp.status_code == 404
