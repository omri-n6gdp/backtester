"""API tests for POST /backtest/walk_forward (Phase 5.2b)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.db.conn import migrate


# A simple strategy whose position size is dispatched from params['size'].
# Bounded to [-1, +1] so the validator accepts it.
PARAM_DISPATCH_CODE = """
import pandas as pd

def strategy(data, params):
    size = float(params.get("size", 1.0))
    if size > 1.0:
        size = 1.0
    if size < -1.0:
        size = -1.0
    return pd.Series(size, index=data.index)
"""


def _fake_get_prices(ticker, timeframe, start, end):
    """600 business days starting 2010-01-01 with positive-drift GBM."""
    n = 600
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0006, 0.012, n)
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
    data_root = tmp_path / "btdata"
    db_dir = data_root / "results"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "results.db"
    strategies_dir = data_root / "strategies"
    strategies_dir.mkdir(parents=True)

    monkeypatch.setattr("backend.storage.DATA_ROOT", data_root, raising=False)
    monkeypatch.setattr("backend.storage.DB_DIR", db_dir, raising=False)
    monkeypatch.setattr("backend.storage.DB_PATH", db_path, raising=False)
    monkeypatch.setattr("backend.storage.STRATEGIES_DIR", strategies_dir, raising=False)
    monkeypatch.setattr("backend.api.routes.STRATEGIES_DIR", strategies_dir, raising=False)
    monkeypatch.setattr("backend.api.routes.get_prices", _fake_get_prices, raising=False)

    migrate()
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/strategies",
            json={
                "name": "Param Dispatch",
                "description": "size from params",
                "code": PARAM_DISPATCH_CODE,
            },
        )
        assert resp.status_code == 200, resp.text
        strategy_id = resp.json()["strategy_id"]
        yield client, strategy_id


def _wf_payload(strategy_id, **overrides):
    payload = {
        "strategy_id": strategy_id,
        "ticker": "SPY",
        "timeframe": "1d",
        "start": "2010-01-01",
        "end": "2013-01-01",
        "param_grid": {"size": [0.5, 1.0]},
        "train_bars": 200,
        "test_bars": 100,
    }
    payload.update(overrides)
    return payload


# --- happy paths ----------------------------------------------------------


def test_walk_forward_with_dict_grid_returns_full_response(client_with_strategy):
    client, strategy_id = client_with_strategy
    resp = client.post("/backtest/walk_forward", json=_wf_payload(strategy_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # 600 bars, train=200, test=100 -> 4 windows; grid has 2 combos.
    assert body["n_windows"] == 4
    assert body["n_combinations"] == 2
    assert body["selection_metric"] == "sharpe"
    assert body["higher_is_better"] is True

    assert len(body["windows"]) == 4
    for w in body["windows"]:
        assert w["n_train_bars"] == 200
        assert w["n_test_bars"] == 100
        assert "size" in w["selected_params"]
        # Each metrics dict has the full MetricsBundle shape.
        assert "sharpe" in w["is_metrics"]
        assert "sharpe" in w["oos_metrics"]

    # Concatenated OOS curve is 4 windows x 100 bars = 400 points.
    assert len(body["oos_timestamps"]) == 400
    assert len(body["oos_equity"]) == 400
    assert len(body["oos_drawdown"]) == 400
    assert len(body["oos_returns"]) == 400


def test_walk_forward_with_list_grid(client_with_strategy):
    client, strategy_id = client_with_strategy
    payload = _wf_payload(
        strategy_id,
        param_grid=[{"size": 0.25}, {"size": 0.75}, {"size": 1.0}],
    )
    resp = client.post("/backtest/walk_forward", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_combinations"] == 3
    selected = {w["selected_params"]["size"] for w in body["windows"]}
    # Selected sizes must come from the grid.
    assert selected.issubset({0.25, 0.75, 1.0})


def test_param_stability_counts_sum_to_n_windows(client_with_strategy):
    client, strategy_id = client_with_strategy
    resp = client.post("/backtest/walk_forward", json=_wf_payload(strategy_id))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    n_windows = body["n_windows"]
    stability = body["param_stability"]
    assert "size" in stability
    counts = stability["size"]["counts"]
    assert sum(counts.values()) == n_windows
    # Stringified keys (JSON requirement).
    for k in counts:
        assert isinstance(k, str)


def test_selection_metric_configurable(client_with_strategy):
    """selection_metric='annualized_vol', higher_is_better=False -> always picks lowest |size|."""
    client, strategy_id = client_with_strategy
    payload = _wf_payload(
        strategy_id,
        param_grid={"size": [0.2, 0.5, 1.0]},
        selection_metric="annualized_vol",
        higher_is_better=False,
    )
    resp = client.post("/backtest/walk_forward", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Vol scales linearly with |position|, so 0.2 always wins.
    for w in body["windows"]:
        assert w["selected_params"]["size"] == 0.2


# --- error cases ----------------------------------------------------------


def test_walk_forward_invalid_selection_metric_returns_400(client_with_strategy):
    client, strategy_id = client_with_strategy
    payload = _wf_payload(strategy_id, selection_metric="not_a_metric")
    resp = client.post("/backtest/walk_forward", json=payload)
    assert resp.status_code == 400, resp.text
    assert "selection_metric" in resp.json()["detail"].lower()


def test_walk_forward_insufficient_bars_returns_400(client_with_strategy):
    client, strategy_id = client_with_strategy
    payload = _wf_payload(strategy_id, train_bars=500, test_bars=200)  # 700 > 600 bars
    resp = client.post("/backtest/walk_forward", json=payload)
    assert resp.status_code == 400, resp.text


def test_walk_forward_unknown_strategy_returns_404(client_with_strategy):
    client, _ = client_with_strategy
    payload = _wf_payload("0" * 32)
    resp = client.post("/backtest/walk_forward", json=payload)
    assert resp.status_code == 404, resp.text


def test_walk_forward_empty_grid_returns_400(client_with_strategy):
    client, strategy_id = client_with_strategy
    payload = _wf_payload(strategy_id, param_grid={})
    resp = client.post("/backtest/walk_forward", json=payload)
    assert resp.status_code == 400, resp.text
