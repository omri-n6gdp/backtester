"""
Unit tests for backend/engine/robustness/heatmaps.py.

The module is pure (no engine dependency), so tests use plain dicts for
the `metrics` argument and plain ints/strings for parameter values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from backend.engine.robustness.heatmaps import (
    HeatmapResult,
    _aggregate,
    _extract_metric,
    build_heatmap,
)


@dataclass
class _StubMetrics:
    sharpe: float
    cagr: float


def _sweep(grid_data: dict[tuple, float], metric_name: str = "sharpe") -> list[tuple[dict, dict]]:
    """Helper: turn {(short, long): sharpe_value, ...} into a sweep result list."""
    out = []
    for (short, long), val in grid_data.items():
        out.append(({"short": short, "long": long}, {metric_name: val}))
    return out


# ---------- helpers ---------------------------------------------------------

def test_extract_metric_from_dict() -> None:
    assert _extract_metric({"sharpe": 0.7, "cagr": 0.09}, "sharpe") == 0.7


def test_extract_metric_from_dict_none_becomes_nan() -> None:
    out = _extract_metric({"sharpe": None}, "sharpe")
    assert math.isnan(out)


def test_extract_metric_from_dataclass() -> None:
    m = _StubMetrics(sharpe=1.2, cagr=0.15)
    assert _extract_metric(m, "sharpe") == 1.2


def test_extract_metric_missing_key_raises() -> None:
    with pytest.raises(KeyError, match="not found"):
        _extract_metric({"sharpe": 0.5}, "calmar")


def test_extract_metric_missing_attr_raises() -> None:
    m = _StubMetrics(sharpe=1.0, cagr=0.1)
    with pytest.raises(KeyError, match="not found"):
        _extract_metric(m, "calmar")


def test_extract_metric_bad_type_raises() -> None:
    with pytest.raises(TypeError, match="dataclass or dict"):
        _extract_metric([1, 2, 3], "sharpe")


def test_aggregate_strategies() -> None:
    vals = [1.0, 2.0, 3.0]
    assert _aggregate(vals, "mean") == 2.0
    assert _aggregate(vals, "median") == 2.0
    assert _aggregate(vals, "min") == 1.0
    assert _aggregate(vals, "max") == 3.0
    assert _aggregate(vals, "last") == 3.0


def test_aggregate_mean_skips_nan() -> None:
    vals = [1.0, float("nan"), 3.0]
    assert _aggregate(vals, "mean") == 2.0


def test_aggregate_all_nan_returns_nan() -> None:
    vals = [float("nan"), float("nan")]
    assert math.isnan(_aggregate(vals, "mean"))


def test_aggregate_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown agg"):
        _aggregate([1.0], "geometric_mean")


# ---------- build_heatmap basic shape ---------------------------------------

def test_build_heatmap_full_grid() -> None:
    sweep = _sweep({
        (10, 100): 0.5, (10, 150): 0.6, (10, 200): 0.7,
        (20, 100): 0.4, (20, 150): 0.55, (20, 200): 0.65,
        (50, 100): 0.3, (50, 150): 0.45, (50, 200): 0.6,
    })
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")
    assert h.x_values == [10, 20, 50]
    assert h.y_values == [100, 150, 200]
    assert h.x_param == "short" and h.y_param == "long"
    assert h.metric_name == "sharpe"
    assert h.n_cells_filled == 9
    assert h.n_cells_total == 9
    # Row 0 is y=100 (long=100); columns are x=10, 20, 50
    assert h.grid[0] == [0.5, 0.4, 0.3]
    assert h.grid[2] == [0.7, 0.65, 0.6]
    assert not any(any(row) for row in h.missing_mask)


def test_build_heatmap_axes_sorted_ascending() -> None:
    sweep = _sweep({
        (50, 200): 1.0, (10, 100): 2.0, (20, 150): 3.0,
    })
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")
    assert h.x_values == [10, 20, 50]
    assert h.y_values == [100, 150, 200]


def test_build_heatmap_missing_cells_marked() -> None:
    sweep = _sweep({
        (10, 100): 0.5,
        (50, 200): 0.7,
    })
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")
    assert h.x_values == [10, 50]
    assert h.y_values == [100, 200]
    assert h.n_cells_filled == 2
    assert h.n_cells_total == 4
    assert not h.missing_mask[0][0]
    assert not h.missing_mask[1][1]
    assert h.missing_mask[0][1]
    assert h.missing_mask[1][0]
    assert math.isnan(h.grid[0][1])
    assert math.isnan(h.grid[1][0])


def test_build_heatmap_works_with_dataclass_metrics() -> None:
    sweep = [
        ({"short": 10, "long": 100}, _StubMetrics(sharpe=0.5, cagr=0.08)),
        ({"short": 20, "long": 100}, _StubMetrics(sharpe=0.6, cagr=0.09)),
    ]
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")
    assert h.grid[0] == [0.5, 0.6]
    h2 = build_heatmap(sweep, x_param="short", y_param="long", metric_name="cagr")
    assert h2.grid[0] == [0.08, 0.09]


def test_build_heatmap_single_point() -> None:
    sweep = _sweep({(10, 100): 0.5})
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")
    assert h.grid == [[0.5]]
    assert h.n_cells_filled == 1


# ---------- duplicates + agg ------------------------------------------------

def test_build_heatmap_default_agg_is_last() -> None:
    sweep = [
        ({"short": 10, "long": 100}, {"sharpe": 0.5}),
        ({"short": 10, "long": 100}, {"sharpe": 0.7}),
    ]
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")
    assert h.grid[0][0] == 0.7


def test_build_heatmap_agg_mean() -> None:
    sweep = [
        ({"short": 10, "long": 100}, {"sharpe": 0.4}),
        ({"short": 10, "long": 100}, {"sharpe": 0.6}),
    ]
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe", agg="mean")
    assert h.grid[0][0] == 0.5


def test_build_heatmap_agg_max() -> None:
    sweep = [
        ({"short": 10, "long": 100}, {"sharpe": 0.4}),
        ({"short": 10, "long": 100}, {"sharpe": 0.6}),
    ]
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe", agg="max")
    assert h.grid[0][0] == 0.6


def test_build_heatmap_agg_min() -> None:
    sweep = [
        ({"short": 10, "long": 100}, {"sharpe": 0.4}),
        ({"short": 10, "long": 100}, {"sharpe": 0.6}),
    ]
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe", agg="min")
    assert h.grid[0][0] == 0.4


# ---------- validation ------------------------------------------------------

def test_same_x_and_y_param_raises() -> None:
    sweep = _sweep({(10, 100): 0.5})
    with pytest.raises(ValueError, match="must differ"):
        build_heatmap(sweep, x_param="short", y_param="short", metric_name="sharpe")


def test_unknown_agg_raises() -> None:
    sweep = _sweep({(10, 100): 0.5})
    with pytest.raises(ValueError, match="agg must be"):
        build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe", agg="rms")


def test_missing_x_param_in_entry_raises() -> None:
    sweep = [
        ({"short": 10, "long": 100}, {"sharpe": 0.5}),
        ({"long": 100}, {"sharpe": 0.6}),
    ]
    with pytest.raises(KeyError, match="missing x_param"):
        build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")


def test_missing_y_param_in_entry_raises() -> None:
    sweep = [
        ({"short": 10}, {"sharpe": 0.5}),
    ]
    with pytest.raises(KeyError, match="missing y_param"):
        build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")


def test_missing_metric_key_raises() -> None:
    sweep = [
        ({"short": 10, "long": 100}, {"cagr": 0.05}),
    ]
    with pytest.raises(KeyError, match="not found"):
        build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")


def test_empty_sweep_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_heatmap([], x_param="short", y_param="long", metric_name="sharpe")


def test_non_dict_params_raises() -> None:
    sweep = [
        ([10, 100], {"sharpe": 0.5}),
    ]
    with pytest.raises(TypeError, match="params must be a dict"):
        build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")


def test_mixed_type_axis_values_raises() -> None:
    """Mixed types on an axis (int + str) can't be sorted together → ValueError."""
    sweep = [
        ({"short": 10, "long": "fast"}, {"sharpe": 0.5}),
        ({"short": 20, "long": 100}, {"sharpe": 0.6}),
    ]
    with pytest.raises(ValueError, match="cannot be sorted"):
        build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")


# ---------- float and string parameter axes --------------------------------

def test_float_parameter_axis() -> None:
    sweep = _sweep({
        (0.1, 100): 0.3,
        (0.2, 100): 0.5,
        (0.3, 100): 0.6,
    })
    h = build_heatmap(sweep, x_param="short", y_param="long", metric_name="sharpe")
    assert h.x_values == [0.1, 0.2, 0.3]
    assert h.grid == [[0.3, 0.5, 0.6]]


def test_string_parameter_axis() -> None:
    sweep = [
        ({"mode": "long", "n": 10}, {"sharpe": 0.5}),
        ({"mode": "short", "n": 10}, {"sharpe": 0.4}),
        ({"mode": "long", "n": 20}, {"sharpe": 0.55}),
        ({"mode": "short", "n": 20}, {"sharpe": 0.45}),
    ]
    h = build_heatmap(sweep, x_param="mode", y_param="n", metric_name="sharpe")
    assert h.x_values == ["long", "short"]
    assert h.y_values == [10, 20]
    assert h.grid[0] == [0.5, 0.4]


# ---------- realistic 3x3 SMA grid scenario --------------------------------

def test_smooth_vs_spiky_heatmap_distinction() -> None:
    """Two synthetic heatmaps: one smooth, one spiky. The output should
    expose that difference (e.g. via stdev of cell values). This is the
    diagnostic 5.6 enables in the frontend — at the data-shape level we
    only verify the values come through, not the rendering."""
    smooth = _sweep({
        (10, 100): 0.6, (10, 150): 0.65, (10, 200): 0.7,
        (20, 100): 0.55, (20, 150): 0.6, (20, 200): 0.65,
        (50, 100): 0.5, (50, 150): 0.55, (50, 200): 0.6,
    })
    spiky = _sweep({
        (10, 100): 0.1, (10, 150): 0.1, (10, 200): 0.1,
        (20, 100): 0.1, (20, 150): 2.5, (20, 200): 0.1,
        (50, 100): 0.1, (50, 150): 0.1, (50, 200): 0.1,
    })
    h_smooth = build_heatmap(smooth, x_param="short", y_param="long", metric_name="sharpe")
    h_spiky = build_heatmap(spiky, x_param="short", y_param="long", metric_name="sharpe")

    import numpy as np
    arr_smooth = np.array(h_smooth.grid)
    arr_spiky = np.array(h_spiky.grid)
    assert arr_smooth.std() < arr_spiky.std()
    assert arr_spiky.max() / np.median(arr_spiky) > 10
