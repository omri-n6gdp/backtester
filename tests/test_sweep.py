"""
Tests for backend/engine/robustness/sweep.py and POST /backtest/sweep.
27 tests covering helpers, run_sweep integration, and the API endpoint.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.engine.costs import DEFAULT_COST_CONFIG
from backend.engine.metrics import MetricsBundle
from backend.engine.robustness.sweep import (
    ComboResult,
    HeatmapRequest,
    _empirical_trial_variance,
    _select_best,
    expand_grid,
    is_metrics_field,
    run_sweep,
)
from tests.conftest import install_fake_prices, post_strategy


# ---- shared fixtures and helpers ----------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": 1e6},
        index=idx,
    )


def _sma_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    short = int(params.get("short", 10))
    long_ = int(params.get("long", 30))
    sig = (data["close"].rolling(short).mean() > data["close"].rolling(long_).mean()).astype(float)
    return sig.fillna(0.0)


def _const_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    size = float(params.get("size", 0.5))
    return pd.Series(size, index=data.index)


# ---- expand_grid ---------------------------------------------------------

def test_expand_grid_dict_cartesian():
    grid = {"a": [1, 2], "b": [10, 20]}
    combos = expand_grid(grid)
    assert len(combos) == 4
    # Every combo should have both keys.
    assert all("a" in c and "b" in c for c in combos)
    # The Cartesian product should cover all (a, b) pairs.
    pairs = {(c["a"], c["b"]) for c in combos}
    assert pairs == {(1, 10), (1, 20), (2, 10), (2, 20)}


def test_expand_grid_list_passthrough():
    explicit = [{"a": 1, "b": 10}, {"a": 2, "b": 99}]
    combos = expand_grid(explicit)
    assert combos == explicit
    # Should be copies, not the same objects.
    assert combos[0] is not explicit[0]


def test_expand_grid_empty_dict_raises():
    with pytest.raises(ValueError, match="empty"):
        expand_grid({})


def test_expand_grid_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        expand_grid([])


def test_expand_grid_invalid_type_raises():
    with pytest.raises(TypeError):
        expand_grid("not_a_grid")  # type: ignore[arg-type]


# ---- is_metrics_field ---------------------------------------------------

def test_is_metrics_field_valid():
    assert is_metrics_field("sharpe") is True
    assert is_metrics_field("cagr") is True
    assert is_metrics_field("max_drawdown") is True


def test_is_metrics_field_invalid():
    assert is_metrics_field("not_a_real_metric") is False
    assert is_metrics_field("") is False


# ---- _select_best -------------------------------------------------------

def test_select_best_higher_is_better():
    scores = [0.5, 1.2, 0.8, float("nan")]
    assert _select_best(scores, higher_is_better=True) == 1


def test_select_best_lower_is_better():
    scores = [0.5, 1.2, 0.3, float("nan")]
    assert _select_best(scores, higher_is_better=False) == 2


def test_select_best_all_nan_returns_none():
    scores = [float("nan"), float("nan")]
    assert _select_best(scores, higher_is_better=True) is None


# ---- _empirical_trial_variance ------------------------------------------

def _make_combo(sharpe: float) -> ComboResult:
    import dataclasses
    fields = {f.name: 0.0 for f in dataclasses.fields(MetricsBundle)}
    fields["sharpe"] = sharpe
    return ComboResult(index=0, params={}, metrics=MetricsBundle(**fields), score=sharpe)


def test_empirical_trial_variance_normal():
    combos = [_make_combo(s) for s in [0.5, 1.0, 1.5]]
    v = _empirical_trial_variance(combos, bars_per_year=252)
    assert v > 0


def test_empirical_trial_variance_all_nan_returns_zero():
    combos = [_make_combo(float("nan")), _make_combo(float("nan"))]
    v = _empirical_trial_variance(combos, bars_per_year=252)
    assert v == 0.0


def test_empirical_trial_variance_single_entry_returns_zero():
    combos = [_make_combo(1.0)]
    v = _empirical_trial_variance(combos, bars_per_year=252)
    assert v == 0.0


# ---- run_sweep integration ----------------------------------------------

DATA = _make_ohlcv()


def test_run_sweep_returns_correct_n_combinations():
    grid = {"short": [5, 10], "long": [20, 30]}
    result = run_sweep(_sma_strategy, DATA, grid)
    assert result.n_combinations == 4
    assert len(result.combos) == 4


def test_run_sweep_best_index_is_valid():
    grid = {"short": [5, 10], "long": [20, 30]}
    result = run_sweep(_sma_strategy, DATA, grid)
    assert result.best_index is not None
    assert 0 <= result.best_index < 4
    assert result.best_combo is not None


def test_run_sweep_winner_has_full_result():
    grid = {"short": [5, 10], "long": [20, 30]}
    result = run_sweep(_sma_strategy, DATA, grid)
    assert result.best_combo is not None
    assert result.best_combo.full_result is not None


def test_run_sweep_non_winners_have_no_full_result():
    grid = {"short": [5, 10], "long": [20, 30]}
    result = run_sweep(_sma_strategy, DATA, grid)
    non_winners = [c for c in result.combos if c.index != result.best_index]
    assert all(c.full_result is None for c in non_winners)


def test_run_sweep_failed_combo_surfaces_as_error_artifact():
    """A combo that raises mid-backtest must appear with error set, not be silently dropped."""
    call_count = {"n": 0}

    def flaky(data: pd.DataFrame, params: dict) -> pd.Series:
        call_count["n"] += 1
        if params.get("size") == 0.9:
            raise RuntimeError("deliberate failure")
        return pd.Series(float(params["size"]), index=data.index)

    grid = {"size": [0.5, 0.9]}
    result = run_sweep(flaky, DATA, grid)
    assert result.n_combinations == 2
    errors = [c for c in result.combos if c.error is not None]
    oks = [c for c in result.combos if c.error is None]
    assert len(errors) == 1
    assert len(oks) == 1
    assert "RuntimeError" in errors[0].error


def test_run_sweep_base_params_merged():
    """base_params should merge into every combo; combo values win on clash."""
    seen_params = []

    def recorder(data: pd.DataFrame, params: dict) -> pd.Series:
        seen_params.append(dict(params))
        size = min(max(float(params.get("size", 0.5)), -1.0), 1.0)
        return pd.Series(size, index=data.index)

    grid = {"size": [0.3, 0.6]}
    run_sweep(recorder, DATA, grid, base_params={"extra": 99})
    assert all(p.get("extra") == 99 for p in seen_params)
    assert {p["size"] for p in seen_params} == {0.3, 0.6}


def test_run_sweep_param_keys_contains_all_swept_params():
    grid = {"short": [5, 10], "long": [20, 30]}
    result = run_sweep(_sma_strategy, DATA, grid)
    assert "short" in result.param_keys
    assert "long" in result.param_keys


def test_run_sweep_heatmap_returned_when_requested():
    grid = {"short": [5, 10], "long": [20, 30]}
    hm_req = [HeatmapRequest(x_param="short", y_param="long", metric_name="sharpe")]
    result = run_sweep(_sma_strategy, DATA, grid, heatmap_requests=hm_req)
    assert len(result.heatmaps) == 1
    hm = result.heatmaps[0]
    assert hm.x_param == "short"
    assert hm.y_param == "long"
    assert hm.metric_name == "sharpe"


def test_run_sweep_heatmap_shape_correct():
    grid = {"short": [5, 10], "long": [20, 30]}
    hm_req = [HeatmapRequest(x_param="short", y_param="long", metric_name="sharpe")]
    result = run_sweep(_sma_strategy, DATA, grid, heatmap_requests=hm_req)
    hm = result.heatmaps[0]
    # 2 x-values × 2 y-values → 2 rows, 2 cols.
    assert len(hm.x_values) == 2
    assert len(hm.y_values) == 2
    assert len(hm.grid) == 2 and all(len(row) == 2 for row in hm.grid)


def test_run_sweep_deflated_sharpe_included_by_default():
    grid = {"size": [0.3, 0.6]}
    result = run_sweep(_const_strategy, DATA, grid, include_deflated_sharpe=True)
    assert result.deflated_sharpe is not None
    assert result.deflated_sharpe.n_trials == 2


def test_run_sweep_deflated_sharpe_excluded_when_disabled():
    grid = {"size": [0.3, 0.6]}
    result = run_sweep(_const_strategy, DATA, grid, include_deflated_sharpe=False)
    assert result.deflated_sharpe is None


def test_run_sweep_invalid_selection_metric_raises():
    grid = {"size": [0.3, 0.6]}
    with pytest.raises(ValueError, match="selection_metric"):
        run_sweep(_const_strategy, DATA, grid, selection_metric="not_a_field")


def test_run_sweep_invalid_heatmap_metric_raises():
    grid = {"size": [0.3, 0.6]}
    hm_req = [HeatmapRequest(x_param="size", y_param="size", metric_name="not_a_field")]
    with pytest.raises(ValueError, match="heatmap metric_name"):
        run_sweep(_const_strategy, DATA, grid, heatmap_requests=hm_req)


# ---- API endpoint -------------------------------------------------------

SIMPLE_STRATEGY_CODE = """
import pandas as pd

def strategy(data, params):
    size = float(params.get("size", 0.5))
    size = max(-1.0, min(1.0, size))
    return pd.Series(size, index=data.index)
"""


@pytest.fixture
def sweep_client(api_client, monkeypatch):
    install_fake_prices(monkeypatch, lambda *_a, **_kw: _make_ohlcv())
    sid = post_strategy(api_client, "Simple", SIMPLE_STRATEGY_CODE)
    yield api_client, sid


def test_api_sweep_returns_200_with_correct_structure(sweep_client):
    client, strategy_id = sweep_client
    payload = {
        "strategy_id": strategy_id,
        "ticker": "SPY",
        "timeframe": "1d",
        "start": "2015-01-01",
        "end": "2016-01-01",
        "param_grid": {"size": [0.3, 0.6, 0.9]},
        "heatmaps": [],
        "include_deflated_sharpe": True,
    }
    resp = client.post("/backtest/sweep", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_combinations"] == 3
    assert len(body["combos"]) == 3
    assert body["best_index"] is not None
    assert body["deflated_sharpe"] is not None
    assert body["deflated_sharpe"]["n_trials"] == 3
    assert "run_id" in body
    assert isinstance(body["param_keys"], list)
    assert "size" in body["param_keys"]
