"""
Parameter sweep with sweep-level robustness rollup (Phase 5.7).

A `sweep` evaluates every combination in a parameter grid against the
same backtest data and reports:

  1. Per-combo: params, MetricsBundle, and a brief summary.
  2. The "best" combo by a chosen selection_metric.
  3. Sweep-level rollup analyses computed across all combos:
     - Deflated Sharpe using N = number of combos AND the empirical
       variance of per-trial Sharpes (a meaningful improvement over
       the conservative V = 1/T placeholder used when computing DSR
       on a single backtest).
     - Heatmaps for any requested (x_param, y_param, metric) triples.

This module is intentionally narrower than `walk_forward`:

  - Walk-forward optimizes parameters PER ROLLING WINDOW and concatenates
    out-of-sample test windows into one aggregate curve. Useful for
    parameter robustness over time.
  - Sweep evaluates every combo over the SAME data range. Useful for
    visualizing the parameter surface and for accountancy when computing
    a deflated Sharpe with a known trial count.

The two can be combined later (walk-forward over a sweep, or sweep-of-
walk-forwards), but the routes stay separate so each tool answers exactly
one question. This phase wires the sweep route only.

What this module does NOT do
----------------------------
- Per-combo OOS splits, Monte Carlo, regime analyses. Those are owned by
  the route layer (which already calls them on the single-backtest path)
  and applied to the *best* combo only, after sweep completes. Running
  them per combo would multiply runtime by 5x for analyses the user
  probably doesn't need to see on every grid cell.
- Persistence. The route writes runs to SQLite; the engine module
  returns pure data structures.

Look-ahead protection: inherited from `run_backtest`. Each combo calls
into the engine, which applies the `.shift(1)` look-ahead guard. Nothing
in this module ever sees raw_positions before the shift.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ..contract import StrategyFn
from ..costs import DEFAULT_COST_CONFIG, CostConfig
from ..metrics import MetricsBundle, compute_metrics
from ..vectorized import BacktestResult, run_backtest
from .deflated_sharpe import DeflatedSharpeResult, compute_deflated_sharpe
from .heatmaps import HeatmapResult, build_heatmap


ParamGrid = Union[dict, list]


@dataclass
class HeatmapRequest:
    """One (x_param, y_param, metric_name) triple to render."""
    x_param: str
    y_param: str
    metric_name: str
    agg: str = "last"


@dataclass
class ComboResult:
    """Per-combo summary. Heavy fields (equity curves, trades) live on
    `full_result` only when this is the best combo — otherwise None to
    keep response sizes bounded."""
    index: int
    params: dict
    metrics: MetricsBundle
    score: float                       # value of selection_metric, or NaN on failure
    error: str | None = None           # populated when this combo raised
    full_result: BacktestResult | None = None  # only for the best combo


@dataclass
class SweepResult:
    """Top-level output. Pure data — route layer marshals to API schema."""
    combos: list[ComboResult]
    n_combinations: int
    selection_metric: str
    higher_is_better: bool
    best_index: int | None
    best_combo: ComboResult | None
    deflated_sharpe: DeflatedSharpeResult | None
    heatmaps: list[HeatmapResult]
    param_keys: list[str]              # union of param keys across the grid


# ---- helpers (public — see module docstring on duplication vs walk_forward) ---

def expand_grid(grid: ParamGrid) -> list[dict]:
    """Expand a grid dict (cartesian) or list-of-dicts (passthrough) into combos.

    Semantically identical to walk_forward._expand_grid; reimplemented here
    rather than imported across modules to avoid cross-module coupling on
    a leading-underscore name. If/when these duplicate further, lift to a
    shared util module — until then duplication is the lesser evil.
    """
    if isinstance(grid, list):
        if not grid:
            raise ValueError("param_grid is an empty list")
        for i, entry in enumerate(grid):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"param_grid[{i}] must be a dict; got {type(entry).__name__}"
                )
        return [dict(d) for d in grid]

    if isinstance(grid, dict):
        if not grid:
            raise ValueError("param_grid is an empty dict")
        keys = sorted(grid.keys())
        per_key_values: list[list] = []
        for k in keys:
            v = grid[k]
            if isinstance(v, (list, tuple)):
                vlist = list(v)
                if not vlist:
                    raise ValueError(f"param_grid[{k!r}] has empty value list")
                per_key_values.append(vlist)
            else:
                per_key_values.append([v])
        combos: list[dict] = []
        for tup in product(*per_key_values):
            combos.append(dict(zip(keys, tup)))
        return combos

    raise TypeError(
        f"param_grid must be dict or list[dict]; got {type(grid).__name__}"
    )


def is_metrics_field(name: str) -> bool:
    """Is `name` a real field on MetricsBundle?"""
    return name in {f.name for f in dataclasses.fields(MetricsBundle)}


# ---- main entry point ---------------------------------------------------

def run_sweep(
    strategy: StrategyFn,
    data: pd.DataFrame,
    param_grid: ParamGrid,
    *,
    base_params: dict | None = None,
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
    starting_capital: float = 100_000.0,
    bars_per_year: int = 252,
    selection_metric: str = "sharpe",
    higher_is_better: bool = True,
    heatmap_requests: list[HeatmapRequest] | None = None,
    include_deflated_sharpe: bool = True,
) -> SweepResult:
    """Evaluate every combo in `param_grid` against `data` and roll up.

    Args
    ----
    strategy:
        validated StrategyFn (data, params) -> Series of target positions.
    data:
        OHLCV DataFrame.
    param_grid:
        Same shape as walk_forward — dict (cartesian-expanded; scalar
        values OK as singletons) or list of dicts (passthrough).
    base_params:
        Optional dict merged into every combo (combo values win on key
        clashes). Useful when most params are fixed and only a few are
        being swept.
    selection_metric, higher_is_better:
        How to pick the "best" combo. Same convention as walk_forward.
    heatmap_requests:
        Optional list of (x_param, y_param, metric_name) triples. Each
        produces one HeatmapResult in the output. The two named params
        must appear in every combo's params dict; the metric must be a
        MetricsBundle field. Pass None or [] to skip heatmaps.
    include_deflated_sharpe:
        When True (default), compute DSR on the best combo's returns
        using N = number of combos and the EMPIRICAL variance of the
        per-trial Sharpes. When False, returns None for the DSR field.

    Returns
    -------
    SweepResult — pure data; the route layer marshals to the API schema.
    """
    # --- validate -------------------------------------------------------
    if not is_metrics_field(selection_metric):
        valid = sorted(f.name for f in dataclasses.fields(MetricsBundle))
        raise ValueError(
            f"selection_metric {selection_metric!r} is not a field on MetricsBundle. "
            f"Valid: {valid}"
        )
    base = dict(base_params or {})
    combos = expand_grid(param_grid)
    if not combos:
        raise ValueError("param_grid expanded to zero combinations")

    # Validate heatmap requests early so a typo doesn't get discovered
    # only after the sweep loop has finished.
    hm_reqs = list(heatmap_requests or [])
    for hr in hm_reqs:
        if not is_metrics_field(hr.metric_name):
            valid = sorted(f.name for f in dataclasses.fields(MetricsBundle))
            raise ValueError(
                f"heatmap metric_name {hr.metric_name!r} is not a MetricsBundle field. "
                f"Valid: {valid}"
            )

    # --- the sweep loop -------------------------------------------------
    combo_results: list[ComboResult] = []
    keys_seen: set[str] = set()

    for idx, combo in enumerate(combos):
        merged = {**base, **combo}
        keys_seen.update(merged.keys())

        try:
            result = run_backtest(
                strategy, data, merged,
                cost_config=cost_config,
                starting_capital=starting_capital,
                bars_per_year=bars_per_year,
            )
            metrics = compute_metrics(result)
            score = float(getattr(metrics, selection_metric))
            err: str | None = None
        except Exception as e:  # noqa: BLE001
            result = None
            metrics = _zero_metrics_bundle()
            score = float("nan")
            err = f"{type(e).__name__}: {e}"

        combo_results.append(ComboResult(
            index=idx,
            params=dict(merged),
            metrics=metrics,
            score=score,
            error=err,
            # Keep the BacktestResult around — we'll prune to just the
            # best combo's result after we know which one won.
            full_result=result,
        ))

    # --- pick the winner ------------------------------------------------
    best_idx = _select_best(
        [c.score for c in combo_results], higher_is_better
    )
    best_combo = combo_results[best_idx] if best_idx is not None else None

    # Free memory: keep full_result only on the winner.
    for c in combo_results:
        if c.index != (best_idx if best_idx is not None else -1):
            c.full_result = None

    # --- sweep-level rollups --------------------------------------------
    deflated = None
    if include_deflated_sharpe and best_combo is not None and best_combo.full_result is not None:
        v_empirical = _empirical_trial_variance(
            combo_results, bars_per_year=bars_per_year,
        )
        try:
            deflated = compute_deflated_sharpe(
                best_combo.full_result.returns,
                n_trials=len(combos),
                bars_per_year=bars_per_year,
                trial_sharpes_variance=v_empirical,
            )
        except Exception:  # noqa: BLE001
            deflated = None

    heatmaps: list[HeatmapResult] = []
    if hm_reqs:
        # Pair each combo with its metrics for build_heatmap. NaN/failed
        # combos still appear (their metric value will be 0.0 from the
        # zeroed MetricsBundle); we don't want to silently drop them
        # because that would lie about the parameter surface — the user
        # should see "this combo failed" as a visible artifact.
        sweep_pairs = [(c.params, c.metrics) for c in combo_results]
        for hr in hm_reqs:
            heatmaps.append(build_heatmap(
                sweep_pairs,
                x_param=hr.x_param,
                y_param=hr.y_param,
                metric_name=hr.metric_name,
                agg=hr.agg,
            ))

    return SweepResult(
        combos=combo_results,
        n_combinations=len(combos),
        selection_metric=selection_metric,
        higher_is_better=higher_is_better,
        best_index=best_idx,
        best_combo=best_combo,
        deflated_sharpe=deflated,
        heatmaps=heatmaps,
        param_keys=sorted(keys_seen),
    )


# ---- internals ----------------------------------------------------------

def _zero_metrics_bundle() -> MetricsBundle:
    """A 'failure' MetricsBundle used when a combo raises mid-backtest.
    All fields zeroed. Saves callers from None-checks downstream."""
    fields = {f.name: 0 if f.type in (int, "int") else 0.0
              for f in dataclasses.fields(MetricsBundle)}
    return MetricsBundle(**fields)


def _select_best(scores: list[float], higher_is_better: bool) -> Optional[int]:
    """Return the index of the best valid score, or None if all are NaN/None."""
    valid_indices = [
        i for i, s in enumerate(scores)
        if s is not None and not (isinstance(s, float) and np.isnan(s))
    ]
    if not valid_indices:
        return None
    cmp = max if higher_is_better else min
    return cmp(valid_indices, key=lambda i: scores[i])


def _empirical_trial_variance(
    combo_results: list[ComboResult],
    bars_per_year: int,
) -> float:
    """Compute Var(per_bar_Sharpe_i) across all trials i.

    The DSR formula expects per-bar Sharpe variance. We have annualized
    Sharpes in metrics.sharpe; to convert: SR_pb = SR_ann / sqrt(bpy).
    Variance scales by the square: Var(SR_pb) = Var(SR_ann) / bpy.

    NaN-handling: failed combos have score=NaN. We drop them rather than
    treating their absent Sharpe as zero — including failures as zeros
    would understate variance and over-deflate DSR. ddof=1 (sample
    variance) is consistent with how Sharpe itself is computed in the
    metrics module."""
    pb_sharpes = []
    for c in combo_results:
        sr_ann = c.metrics.sharpe
        if sr_ann is None or (isinstance(sr_ann, float) and math.isnan(sr_ann)):
            continue
        pb_sharpes.append(sr_ann / math.sqrt(bars_per_year))

    if len(pb_sharpes) < 2:
        # Can't estimate variance from one observation. Return 0.0, which
        # makes DSR collapse to PSR — the most conservative thing to do
        # when we can't tell whether there was selection bias.
        return 0.0

    return float(np.var(pb_sharpes, ddof=1))
