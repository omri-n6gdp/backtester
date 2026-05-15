"""
API-level tests for POST /backtest/event_driven (Phase 9).
"""
from __future__ import annotations

import pytest

from tests.conftest import install_fake_prices, make_synthetic_ohlcv, post_strategy


# Long-and-hold: always returns +1. With a trailing stop and noisy data,
# we expect at least one stop-hit trade.
LONG_HOLD_CODE = """
import pandas as pd

def strategy(data, params):
    return pd.Series(1.0, index=data.index)
"""


@pytest.fixture
def ed_client(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: make_synthetic_ohlcv(n=400, seed=11))
    sid = post_strategy(api_client, "Long & Hold", LONG_HOLD_CODE)
    return api_client, sid


def _payload(sid, **overrides):
    p = {
        "strategy_id": sid,
        "ticker": "SPY",
        "timeframe": "1d",
        "start": "2010-01-01",
        "end": "2012-01-01",
        "params": {},
        "stop_config": {"trailing_stop_pct": 0.05},
    }
    p.update(overrides)
    return p


def test_event_driven_trailing_stop_produces_stop_hit_trades(ed_client):
    client, sid = ed_client
    resp = client.post("/backtest/event_driven", json=_payload(sid))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # At least one trade must have been closed by the trailing stop.
    reasons = [t.get("exit_reason") for t in body["trades"]]
    assert "trailing_stop" in reasons, f"got exit reasons {reasons}"


def test_event_driven_negative_pct_returns_400(ed_client):
    client, sid = ed_client
    resp = client.post(
        "/backtest/event_driven",
        json=_payload(sid, stop_config={"stop_loss_pct": -0.1}),
    )
    assert resp.status_code == 400, resp.text


def test_event_driven_run_persists_and_round_trips_via_get(ed_client):
    client, sid = ed_client
    resp = client.post("/backtest/event_driven", json=_payload(sid))
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    got = client.get(f"/runs/{run_id}").json()
    assert got["run_id"] == run_id
    assert len(got["trades"]) == len(resp.json()["trades"])
    # exit_reason is part of the Trade dataclass and should survive
    # JSON persistence.
    persisted_reasons = [t.get("exit_reason") for t in got["trades"]]
    assert "trailing_stop" in persisted_reasons
