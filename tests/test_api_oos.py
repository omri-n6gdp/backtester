"""API tests for the OOS split feature added in Phase 5.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.db.conn import migrate


BUY_AND_HOLD_CODE = """
import pandas as pd

def strategy(data, params):
    return pd.Series(1.0, index=data.index)
"""


def _fake_get_prices(ticker, timeframe, start, end):
    """Synthetic OHLCV — 500 business days from 2010-01-01."""
    n = 500
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2010-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )


@pytest.fixture
def client_with_strategy(tmp_path, monkeypatch):
    """TestClient against a tmp DB, with a saved buy-and-hold strategy."""
    data_root = tmp_path / "btdata"
    db_dir = data_root / "results"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "results.db"
    strategies_dir = data_root / "strategies"
    strategies_dir.mkdir(parents=True)

    # Patch storage paths in their canonical home and any module that
    # imported them by name at module load (routes is the one we know).
    monkeypatch.setattr("backend.storage.DATA_ROOT", data_root, raising=False)
    monkeypatch.setattr("backend.storage.DB_DIR", db_dir, raising=False)
    monkeypatch.setattr("backend.storage.DB_PATH", db_path, raising=False)
    monkeypatch.setattr("backend.storage.STRATEGIES_DIR", strategies_dir, raising=False)
    monkeypatch.setattr("backend.api.routes.STRATEGIES_DIR", strategies_dir, raising=False)
    monkeypatch.setattr("backend.api.routes.get_prices", _fake_get_prices, raising=False)

    # Run migrations directly; FastAPI startup hook is unreliable under TestClient.
    migrate()

    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/strategies",
            json={
                "name": "Buy and Hold",
                "description": "always long",
                "code": BUY_AND_HOLD_CODE,
            },
        )
        assert resp.status_code == 200, resp.text
        strategy_id = resp.json()["strategy_id"]
        yield client, strategy_id


def _backtest_payload(strategy_id, **overrides):
    payload = {
        "strategy_id": strategy_id,
        "ticker": "SPY",
        "timeframe": "1d",
        "start": "2010-01-01",
        "end": "2012-01-01",
        "params": {},
    }
    payload.update(overrides)
    return payload


def test_backtest_without_oos_split_omits_oos_field(client_with_strategy):
    client, strategy_id = client_with_strategy
    resp = client.post("/backtest", json=_backtest_payload(strategy_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("oos") is None


def test_backtest_with_fraction_oos_split_returns_both_windows(client_with_strategy):
    client, strategy_id = client_with_strategy
    resp = client.post("/backtest", json=_backtest_payload(strategy_id, oos_split=0.7))
    assert resp.status_code == 200, resp.text
    oos = resp.json()["oos"]
    assert oos is not None
    assert oos["in_sample"]["label"] == "in_sample"
    assert oos["out_of_sample"]["label"] == "out_of_sample"
    assert oos["in_sample"]["n_bars"] + oos["out_of_sample"]["n_bars"] == 500
    assert oos["split_index"] == 350  # int(500 * 0.7)
    is_metrics = oos["in_sample"]["metrics"]
    oos_metrics = oos["out_of_sample"]["metrics"]
    assert "sharpe" in is_metrics
    assert "cagr" in oos_metrics
    assert "n_trades" in is_metrics


def test_backtest_with_date_oos_split(client_with_strategy):
    client, strategy_id = client_with_strategy
    resp = client.post(
        "/backtest", json=_backtest_payload(strategy_id, oos_split="2010-12-01")
    )
    assert resp.status_code == 200, resp.text
    oos = resp.json()["oos"]
    assert oos is not None
    assert oos["split_timestamp"] >= "2010-12-01"
    assert oos["in_sample"]["n_bars"] > 0
    assert oos["out_of_sample"]["n_bars"] > 0
    assert oos["in_sample"]["n_bars"] + oos["out_of_sample"]["n_bars"] == 500


def test_backtest_with_invalid_oos_split_returns_400(client_with_strategy):
    client, strategy_id = client_with_strategy
    # Out-of-range fraction.
    resp = client.post(
        "/backtest", json=_backtest_payload(strategy_id, oos_split=1.5)
    )
    assert resp.status_code == 400, resp.text
    assert "oos_split" in resp.json()["detail"].lower()
    # Date before all data.
    resp = client.post(
        "/backtest", json=_backtest_payload(strategy_id, oos_split="1900-01-01")
    )
    assert resp.status_code == 400, resp.text


def test_compound_invariant_holds_through_api(client_with_strategy):
    """(1 + IS_total) * (1 + OOS_total) * (1 + r_split) == (1 + Full_total).

    `total_return` is equity[-1]/equity[0] - 1, which silently drops each
    window's first-bar return from the ratio. The full backtest's first bar
    is r_0; the OOS window's first bar is r_split. These are different bars,
    so the bare two-factor identity does NOT hold and the missing factor is
    exactly (1 + r_split). The engine-level test in tests/test_oos.py uses
    raw equity factors and verifies the simpler two-factor identity directly.
    """
    client, strategy_id = client_with_strategy
    resp = client.post("/backtest", json=_backtest_payload(strategy_id, oos_split=0.5))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    split_index = body["oos"]["split_index"]
    full_total = body["metrics"]["total_return"]
    is_total = body["oos"]["in_sample"]["metrics"]["total_return"]
    oos_total = body["oos"]["out_of_sample"]["metrics"]["total_return"]
    r_at_split = body["returns"][split_index]
    lhs = (1 + is_total) * (1 + oos_total) * (1 + r_at_split)
    rhs = 1 + full_total
    assert lhs == pytest.approx(rhs, rel=1e-10)
