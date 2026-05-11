"""
Parameter heatmaps.

Given a 2D parameter sweep (list of (params, metrics) pairs), produce a
grid of any chosen metric across two parameter axes. The grid is what
the frontend renders as a color-mapped 2D plot.

Why this matters
----------------
A smooth heatmap — where adjacent parameter combos produce similar
metrics — suggests the strategy has a real edge that doesn't depend on
finding exactly the right knob setting. A spiky heatmap — where one
combo is great and its neighbors are terrible — is a strong overfitting
signal: small parameter perturbations should not change a real edge by
much. This module surfaces the *shape* of the parameter-to-metric
surface in a form the eye can scan.

Inputs
------
- `sweep_results`: list of (params: dict, metrics: MetricsBundle | dict)
  pairs. One entry per parameter combination. Order doesn't matter.
- `x_param`, `y_param`: names of the two parameter keys to span. Both
  must appear in every params dict.
- `metric_name`: which metric field to color by (e.g. 'sharpe', 'cagr').
- `agg`: how to collapse duplicate (x, y) cells. By default, callers run
  one combo per cell, but if the same (x, y) appears multiple times
  (e.g. averaged across assets), 'mean' / 'median' / 'min' / 'max'
  / 'last' control the aggregation.

This module is pure: no I/O, no engine calls. The caller is responsible
for running the sweep and feeding it in. That separation is intentional
— 5.7's sweep route owns the sweep loop; this module only shapes the
data for display.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

SUPPORTED_AGGS = ("mean", "median", "min", "max", "last")


@dataclass
class HeatmapResult:
    """2D grid of one metric across two parameter axes.

    The grid is shape (n_y, n_x); row i corresponds to y_values[i] and
    column j corresponds to x_values[j]. Missing cells (parameter combos
    not present in the sweep) are float('nan') in `grid` and True in
    `missing_mask`. The frontend should render missing cells with a
    distinct color (e.g. grey).

    `metric_name` is included so the consumer can label the colorbar
    without a separate roundtrip.
    """
    x_param: str
    y_param: str
    metric_name: str
    x_values: list[Any]              # column labels, sorted ascending
    y_values: list[Any]              # row labels, sorted ascending
    grid: list[list[float]]          # [n_y][n_x], NaN where missing
    missing_mask: list[list[bool]]   # [n_y][n_x], True where missing
    agg: str
    n_cells_filled: int
    n_cells_total: int


def _extract_metric(metrics: Any, metric_name: str) -> float:
    """Pull a metric value from a MetricsBundle dataclass OR a dict.
    Raises KeyError with a useful message if not found."""
    if dataclasses.is_dataclass(metrics):
        if not hasattr(metrics, metric_name):
            raise KeyError(
                f"metric {metric_name!r} not found on {type(metrics).__name__}; "
                f"available: {[f.name for f in dataclasses.fields(metrics)]}"
            )
        return float(getattr(metrics, metric_name))
    if isinstance(metrics, dict):
        if metric_name not in metrics:
            raise KeyError(
                f"metric {metric_name!r} not found in metrics dict; "
                f"available keys: {sorted(metrics.keys())}"
            )
        val = metrics[metric_name]
        if val is None:
            return float("nan")
        return float(val)
    raise TypeError(
        f"metrics must be a dataclass or dict, got {type(metrics).__name__}"
    )


def _aggregate(values: list[float], agg: str) -> float:
    """Collapse a list of values per the agg strategy. Treats NaN inputs
    as missing for mean/median, included as-is for min/max/last."""
    arr = np.array(values, dtype=float)
    if agg == "mean":
        with np.errstate(all="ignore"):
            val = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")
        return val
    if agg == "median":
        with np.errstate(all="ignore"):
            val = float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else float("nan")
        return val
    if agg == "min":
        return float(np.nanmin(arr)) if np.any(~np.isnan(arr)) else float("nan")
    if agg == "max":
        return float(np.nanmax(arr)) if np.any(~np.isnan(arr)) else float("nan")
    if agg == "last":
        return float(arr[-1])
    raise ValueError(f"unknown agg {agg!r}; expected one of {SUPPORTED_AGGS}")


def build_heatmap(
    sweep_results: Iterable[tuple[dict[str, Any], Any]],
    x_param: str,
    y_param: str,
    metric_name: str,
    agg: str = "last",
) -> HeatmapResult:
    """Build a 2D heatmap grid from sweep results.

    Args
    ----
    sweep_results: iterable of (params, metrics) pairs.
    x_param, y_param: which two parameter keys to span. Must be distinct.
        Both keys must appear in EVERY params dict (we raise on the first
        missing one). The grid is sorted ascending on both axes using
        Python's natural ordering, so int axes work cleanly and string
        axes sort lexicographically.
    metric_name: which metric field to color by. The value at each cell
        is `metrics.metric_name` (dataclass) or `metrics[metric_name]`
        (dict). None values become NaN.
    agg: how to collapse if the same (x, y) appears more than once.

    Returns
    -------
    HeatmapResult with serializable nested lists. NaN in `grid` is the
    canonical missing-cell sentinel; `missing_mask` carries the same
    info in boolean form (JSON-safe — NaN is not).
    """
    if x_param == y_param:
        raise ValueError(f"x_param and y_param must differ; got {x_param!r}")
    if agg not in SUPPORTED_AGGS:
        raise ValueError(f"agg must be one of {SUPPORTED_AGGS}, got {agg!r}")

    # Bucket by (x, y) to handle duplicates per the agg strategy.
    buckets: dict[tuple[Any, Any], list[float]] = {}
    x_seen: set[Any] = set()
    y_seen: set[Any] = set()

    n_entries = 0
    for params, metrics in sweep_results:
        n_entries += 1
        if not isinstance(params, dict):
            raise TypeError(
                f"each sweep entry's params must be a dict; got {type(params).__name__}"
            )
        if x_param not in params:
            raise KeyError(
                f"sweep entry {n_entries} missing x_param {x_param!r}; "
                f"available keys: {sorted(params.keys())}"
            )
        if y_param not in params:
            raise KeyError(
                f"sweep entry {n_entries} missing y_param {y_param!r}; "
                f"available keys: {sorted(params.keys())}"
            )
        x = params[x_param]
        y = params[y_param]
        x_seen.add(x)
        y_seen.add(y)
        val = _extract_metric(metrics, metric_name)
        buckets.setdefault((x, y), []).append(val)

    if n_entries == 0:
        raise ValueError("sweep_results is empty; cannot build a heatmap")

    # Sort axes ascending. Mixed types (int + str + None) would crash
    # Python's default sort with a TypeError; we let that propagate so
    # the caller knows the input is malformed.
    try:
        x_values = sorted(x_seen)
        y_values = sorted(y_seen)
    except TypeError as e:
        raise ValueError(
            f"{x_param!r} or {y_param!r} contained values that cannot be sorted "
            f"together (mixed types?): {e}"
        ) from e

    x_to_col = {v: j for j, v in enumerate(x_values)}
    y_to_row = {v: i for i, v in enumerate(y_values)}

    n_x = len(x_values)
    n_y = len(y_values)
    grid = [[float("nan")] * n_x for _ in range(n_y)]
    missing = [[True] * n_x for _ in range(n_y)]

    for (x, y), vals in buckets.items():
        i = y_to_row[y]
        j = x_to_col[x]
        agg_val = _aggregate(vals, agg)
        grid[i][j] = agg_val
        if not np.isnan(agg_val):
            missing[i][j] = False

    n_filled = sum(1 for row in missing for cell in row if not cell)
    return HeatmapResult(
        x_param=x_param,
        y_param=y_param,
        metric_name=metric_name,
        x_values=list(x_values),
        y_values=list(y_values),
        grid=grid,
        missing_mask=missing,
        agg=agg,
        n_cells_filled=n_filled,
        n_cells_total=n_x * n_y,
    )
