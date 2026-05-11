"""
HTTP routes. Thin layer over the engine + repo.
"""
from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.api import VERSION
from backend.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    CostConfigModel,
    HealthResponse,
    MonteCarloConfigModel,
    MonteCarloResponseModel,
    DeflatedSharpeResponseModel,
    RegimeAnalysisResponseModel,
    RegimeAxisResponseModel,
    RegimeWindowModel,
    OOSResponseModel,
    ParamStabilityEntry,
    PricesResponse,
    StrategyCreateRequest,
    StrategyCreateResponse,
    StrategyDetail,
    StrategySummary,
    ValidationResultModel,
    WalkForwardRequest,
    WalkForwardResponse,
    WalkForwardWindowModel,
    WindowMetricsModel,
)
from backend.data.prices import get_prices
from backend.db import repo
from backend.db.conn import connect
from backend.engine.costs import CostConfig, DEFAULT_COST_CONFIG
from backend.engine.metrics import compute_metrics
from backend.engine.robustness.oos import WindowResult, split_result
from backend.engine.robustness.monte_carlo import (
    MonteCarloConfig,
    monte_carlo_shuffle,
)
from backend.engine.robustness.deflated_sharpe import compute_deflated_sharpe
from backend.engine.robustness.regimes import (
    RegimeAnalysisResult,
    RegimeAxisResult,
    analyze_regimes,
)
from backend.engine.robustness.walk_forward import WindowReport, walk_forward
from backend.engine.validator import load_validated_strategy, validate_strategy
from backend.engine.vectorized import run_backtest
from backend.storage import STRATEGIES_DIR

# -------- regime helpers --------

def _fetch_regime_close(ticker: str, start: date, end: date, warmup_days: int = 0) -> tuple[Any, str | None]:
    """Fetch a daily close series for the given ticker. Returns (close_series, error).
    Either may be None. `warmup_days` extends the start backward to give
    rolling indicators (e.g. SPY 200d MA) enough history."""
    fetch_start = start - timedelta(days=warmup_days) if warmup_days > 0 else start
    try:
        df = get_prices(ticker=ticker, timeframe="1d", start=fetch_start, end=end)
    except Exception as e:  # noqa: BLE001
        return None, f"{ticker} fetch failed: {type(e).__name__}: {e}"
    if df is None or df.empty:
        return None, f"{ticker} returned no data for range"
    return df["close"], None


def _axis_to_response(axis: RegimeAxisResult) -> RegimeAxisResponseModel:
    """Engine RegimeAxisResult -> API RegimeAxisResponseModel."""
    return RegimeAxisResponseModel(
        name=axis.name,
        n_bars_total=axis.n_bars_total,
        n_bars_unknown=axis.n_bars_unknown,
        windows={
            label: RegimeWindowModel(
                label=w.label,
                n_bars=w.n_bars,
                fraction=w.fraction,
                metrics=_metrics_to_dict(w.metrics),
            )
            for label, w in axis.windows.items()
        },
    )


router = APIRouter()


# -------- helpers --------

def _series_to_list(s: pd.Series) -> list[float | None]:
    """Convert a pd.Series to a JSON-safe list, mapping NaN -> None."""
    return [None if pd.isna(v) else float(v) for v in s.to_list()]


def _index_to_iso(idx: pd.DatetimeIndex) -> list[str]:
    """ISO 8601 UTC strings for a tz-aware DatetimeIndex."""
    return [ts.isoformat() for ts in idx]


def _trade_to_dict(t: Any) -> dict[str, Any]:
    """Serialize a Trade dataclass (or any object) to a JSON-safe dict."""
    if dataclasses.is_dataclass(t):
        d = dataclasses.asdict(t)
    elif hasattr(t, "__dict__"):
        d = dict(vars(t))
    else:
        d = {"repr": repr(t)}
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (pd.Timestamp, datetime)):
            out[k] = v.isoformat()
        elif isinstance(v, float) and pd.isna(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _metrics_to_dict(m: Any) -> dict[str, Any]:
    """Serialize a MetricsBundle (or other dataclass/object) to JSON-safe dict."""
    if dataclasses.is_dataclass(m):
        d = dataclasses.asdict(m)
    elif hasattr(m, "__dict__"):
        d = dict(vars(m))
    else:
        d = {}
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in d.items()}


def _parse_date(s: str) -> date:
    """Parse an ISO-8601 'YYYY-MM-DD' string into a datetime.date."""
    return date.fromisoformat(s)


def _cost_config_from_model(m: CostConfigModel | None) -> CostConfig:
    if m is None:
        return DEFAULT_COST_CONFIG
    return CostConfig(**m.model_dump())


def _validation_result_to_model(vresult: Any) -> ValidationResultModel:
    """
    Adapt a ValidationResult (dataclass or object) to the Pydantic response model.
    Tolerant of small attribute differences (warnings may not exist in older versions).
    """
    return ValidationResultModel(
        ok=bool(getattr(vresult, "ok", False)),
        errors=list(getattr(vresult, "errors", []) or []),
        warnings=list(getattr(vresult, "warnings", []) or []),
        strategy_name=getattr(vresult, "strategy_name", None),
    )


def _window_to_model(w: WindowResult) -> WindowMetricsModel:
    """Convert a WindowResult (engine) to a WindowMetricsModel (API schema)."""
    return WindowMetricsModel(
        label=w.label,
        start=w.start.isoformat(),
        end=w.end.isoformat(),
        n_bars=w.n_bars,
        metrics=_metrics_to_dict(w.metrics),
    )


def _walk_forward_window_to_model(w: WindowReport) -> WalkForwardWindowModel:
    """Convert a WindowReport (engine) to a WalkForwardWindowModel (API schema)."""
    return WalkForwardWindowModel(
        window_index=w.window_index,
        train_start=w.train_start.isoformat(),
        train_end=w.train_end.isoformat(),
        test_start=w.test_start.isoformat(),
        test_end=w.test_end.isoformat(),
        n_train_bars=w.n_train_bars,
        n_test_bars=w.n_test_bars,
        selected_params=dict(w.selected_params),
        is_metrics=_metrics_to_dict(w.is_metrics),
        oos_metrics=_metrics_to_dict(w.oos_metrics),
    )


def _param_stability_to_models(
    stability: dict[str, dict[str, Any]],
) -> dict[str, ParamStabilityEntry]:
    """Convert engine param_stability dict -> API ParamStabilityEntry models.

    JSON requires string keys, so the engine's `counts` (dict keyed by the
    original parameter values: int, float, str, ...) is stringified here.
    """
    out: dict[str, ParamStabilityEntry] = {}
    for key, entry in stability.items():
        raw_counts = entry["counts"]
        str_counts = {str(k): int(v) for k, v in raw_counts.items()}
        out[key] = ParamStabilityEntry(
            counts=str_counts,
            n_distinct=int(entry["n_distinct"]),
            mode=str(entry["mode"]),
        )
    return out


# -------- /health --------

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


# -------- /prices --------

@router.get("/prices", response_model=PricesResponse)
def prices(
    ticker: str = Query(..., min_length=1),
    timeframe: str = Query("1d"),
    start: str = Query(...),
    end: str = Query(...),
) -> PricesResponse:
    try:
        df = get_prices(ticker=ticker, timeframe=timeframe, start=_parse_date(start), end=_parse_date(end))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e!s}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"no data for {ticker} {timeframe} {start}..{end}")

    return PricesResponse(
        ticker=ticker,
        timeframe=timeframe,
        timestamps=_index_to_iso(df.index),
        open=df["open"].astype(float).to_list(),
        high=df["high"].astype(float).to_list(),
        low=df["low"].astype(float).to_list(),
        close=df["close"].astype(float).to_list(),
        volume=df["volume"].astype(float).to_list(),
    )


# -------- /strategies --------

@router.post("/strategies", response_model=StrategyCreateResponse)
def create_strategy(req: StrategyCreateRequest) -> StrategyCreateResponse:
    # The Phase 2 validator returns a ValidationResult; load_validated_strategy
    # returns just the callable (and raises on failure). We use validate_strategy
    # here because we want to surface failures as 400s, not 500s.
    vresult = validate_strategy(req.code)
    if not vresult.ok:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "strategy validation failed",
                "errors": list(vresult.errors),
                "warnings": list(getattr(vresult, "warnings", []) or []),
            },
        )

    strategy_id = uuid.uuid4().hex
    code_path = STRATEGIES_DIR / f"{strategy_id}.py"
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    code_path.write_text(req.code, encoding="utf-8")

    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        repo.insert_strategy(
            conn,
            strategy_id=strategy_id,
            name=req.name,
            description=req.description,
            code_path=str(code_path),
            created_at=created_at,
        )

    return StrategyCreateResponse(
        strategy_id=strategy_id,
        validation_result=_validation_result_to_model(vresult),
    )


@router.get("/strategies", response_model=list[StrategySummary])
def list_strategies() -> list[StrategySummary]:
    with connect() as conn:
        rows = repo.list_strategies(conn, include_inactive=False)
    return [
        StrategySummary(id=r["id"], name=r["name"], description=r["description"], created_at=r["created_at"])
        for r in rows
    ]


@router.get("/strategies/{strategy_id}", response_model=StrategyDetail)
def get_strategy(strategy_id: str) -> StrategyDetail:
    with connect() as conn:
        row = repo.get_strategy(conn, strategy_id)
    if row is None or row["active"] == 0:
        raise HTTPException(status_code=404, detail="strategy not found")
    code_path = Path(row["code_path"])
    if not code_path.exists():
        raise HTTPException(status_code=410, detail="strategy code file missing on disk")
    code = code_path.read_text(encoding="utf-8")
    return StrategyDetail(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=row["created_at"],
        code=code,
    )


@router.delete("/strategies/{strategy_id}")
def delete_strategy(strategy_id: str) -> dict[str, Any]:
    with connect() as conn:
        ok = repo.soft_delete_strategy(conn, strategy_id)
    if not ok:
        raise HTTPException(status_code=404, detail="strategy not found")
    return {"strategy_id": strategy_id, "deleted": True}


# -------- /backtest --------

@router.post("/backtest", response_model=BacktestResponse)
def run_backtest_route(req: BacktestRequest) -> BacktestResponse:
    # 1. Look up the strategy and load its code.
    with connect() as conn:
        row = repo.get_strategy(conn, req.strategy_id)
    if row is None or row["active"] == 0:
        raise HTTPException(status_code=404, detail="strategy not found")
    code_path = Path(row["code_path"])
    if not code_path.exists():
        raise HTTPException(status_code=410, detail="strategy code file missing on disk")
    code = code_path.read_text(encoding="utf-8")

    # Re-validate before running. load_validated_strategy raises ValueError
    # on failure; we convert that to a 409 (stored code no longer valid).
    try:
        fn = load_validated_strategy(code)
    except ValueError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "stored strategy fails revalidation",
                "errors": [str(e)],
            },
        )

    # 2. Fetch prices.
    try:
        data = get_prices(ticker=req.ticker, timeframe=req.timeframe, start=_parse_date(req.start), end=_parse_date(req.end))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e!s}")
    if data.empty:
        raise HTTPException(status_code=404, detail="no data for requested window")

    # 3. Run the engine.
    cost_config = _cost_config_from_model(req.cost_config)
    try:
        result = run_backtest(
            fn,
            data,
            req.params,
            cost_config=cost_config,
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"backtest failed: {e!s}")

    metrics_bundle = compute_metrics(result)
    metrics_dict = _metrics_to_dict(metrics_bundle)

    # 3.5 Optional in-sample / out-of-sample split.
    oos_response: OOSResponseModel | None = None
    if req.oos_split is not None:
        try:
            oos_result = split_result(result, split=req.oos_split)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"invalid oos_split: {e!s}")
        oos_response = OOSResponseModel(
            split_timestamp=oos_result.split_timestamp.isoformat(),
            split_index=oos_result.split_index,
            in_sample=_window_to_model(oos_result.in_sample),
            out_of_sample=_window_to_model(oos_result.out_of_sample),
        )

    # 3.6 Optional Monte Carlo trade-order shuffle.
    mc_response: MonteCarloResponseModel | None = None
    if req.monte_carlo is not None:
        try:
            mc_cfg = MonteCarloConfig(
                n_iterations=req.monte_carlo.n_iterations,
                seed=req.monte_carlo.seed,
                quantiles=tuple(req.monte_carlo.quantiles),
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"invalid monte_carlo config: {e!s}",
            )
        try:
            mc_result = monte_carlo_shuffle(result, mc_cfg)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"monte_carlo failed: {e!s}")
        # Convert NaN percentile to None for JSON safety (NaN is not valid JSON).
        dd_pct = mc_result.original_dd_percentile
        dd_pct_safe = None if dd_pct != dd_pct else dd_pct
        mc_response = MonteCarloResponseModel(
            n_iterations=mc_result.n_iterations,
            n_trades=mc_result.n_trades,
            seed=mc_result.seed,
            quantiles=mc_result.quantiles,
            original_final_equity=mc_result.original_final_equity,
            original_max_drawdown=mc_result.original_max_drawdown,
            original_max_drawdown_trades=mc_result.original_max_drawdown_trades,
            final_equities=mc_result.final_equities,
            max_drawdowns=mc_result.max_drawdowns,
            max_drawdown_durations=mc_result.max_drawdown_durations,
            equity_bands=mc_result.equity_bands,
            drawdown_bands=mc_result.drawdown_bands,
            max_dd_quantiles=mc_result.max_dd_quantiles,
            original_dd_percentile=dd_pct_safe,
        )

    # 3.7 Optional Probabilistic / Deflated Sharpe Ratio.
    ds_response: DeflatedSharpeResponseModel | None = None
    if req.n_trials is not None:
        try:
            ds_result = compute_deflated_sharpe(
                result.returns,
                n_trials=req.n_trials,
                bars_per_year=req.bars_per_year,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"invalid n_trials: {e!s}",
            )
        ds_response = DeflatedSharpeResponseModel(
            n_returns=ds_result.n_returns,
            n_trials=ds_result.n_trials,
            bars_per_year=ds_result.bars_per_year,
            sharpe_annualized=ds_result.sharpe_annualized,
            sharpe_per_bar=ds_result.sharpe_per_bar,
            skew=ds_result.skew,
            excess_kurtosis=ds_result.excess_kurtosis,
            expected_max_sharpe_annualized=ds_result.expected_max_sharpe_annualized,
            psr=ds_result.psr,
            dsr=ds_result.dsr,
        )

    # 3.8 Optional regime conditional analysis.
    regimes_response: RegimeAnalysisResponseModel | None = None
    if req.include_regimes:
        spy_close, spy_err = _fetch_regime_close(
            "SPY", _parse_date(req.start), _parse_date(req.end), warmup_days=400,
        )
        vix_close, vix_err = _fetch_regime_close(
            "^VIX", _parse_date(req.start), _parse_date(req.end), warmup_days=0,
        )
        try:
            regime_result: RegimeAnalysisResult = analyze_regimes(
                result, spy_close=spy_close, vix_close=vix_close,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"regime analysis failed: {e!s}")
        # Combine fetch errors with engine errors. Fetch errors win
        # because if we couldn't fetch, we never even called the engine.
        spy_err_combined = spy_err or regime_result.spy_trend_error
        vix_err_combined = vix_err or regime_result.vix_level_error
        regimes_response = RegimeAnalysisResponseModel(
            spy_trend=_axis_to_response(regime_result.spy_trend) if regime_result.spy_trend else None,
            vix_level=_axis_to_response(regime_result.vix_level) if regime_result.vix_level else None,
            spy_trend_error=spy_err_combined,
            vix_level_error=vix_err_combined,
        )

    # 4. Persist run + curves + metrics + trades.
    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    timestamps = _index_to_iso(result.equity.index)
    equity_list = _series_to_list(result.equity)
    drawdown_list = _series_to_list(result.drawdown)
    returns_list = _series_to_list(result.returns)
    costs_list = _series_to_list(result.costs)
    raw_pos_list = _series_to_list(result.raw_positions)
    pos_list = _series_to_list(result.positions)
    trades_list = [_trade_to_dict(t) for t in result.trades]

    with connect() as conn:
        repo.insert_run(
            conn,
            run_id=run_id,
            strategy_id=req.strategy_id,
            ticker=req.ticker,
            timeframe=req.timeframe,
            start_date=req.start,
            end_date=req.end,
            params_json=json.dumps(req.params),
            cost_config_json=json.dumps(dataclasses.asdict(cost_config)),
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
            created_at=created_at,
        )
        repo.insert_metrics(conn, run_id=run_id, metrics_json=json.dumps(metrics_dict))
        repo.insert_curves(
            conn,
            run_id=run_id,
            timestamps_json=json.dumps(timestamps),
            equity_json=json.dumps(equity_list),
            drawdown_json=json.dumps(drawdown_list),
            returns_json=json.dumps(returns_list),
            costs_json=json.dumps(costs_list),
            raw_positions_json=json.dumps(raw_pos_list),
            positions_json=json.dumps(pos_list),
        )
        repo.insert_trades(conn, run_id=run_id, trades=trades_list)

    # 5. Build summary.
    n_trades = len(trades_list)
    win_rate = metrics_dict.get("win_rate")
    summary = {
        "cagr": metrics_dict.get("cagr"),
        "sharpe": metrics_dict.get("sharpe"),
        "max_drawdown": metrics_dict.get("max_drawdown"),
        "n_trades": n_trades,
        "win_rate": win_rate,
        "final_equity": equity_list[-1] if equity_list else None,
    }

    return BacktestResponse(
        run_id=run_id,
        metrics=metrics_dict,
        summary=summary,
        timestamps=timestamps,
        equity=equity_list,
        drawdown=drawdown_list,
        returns=returns_list,
        costs=costs_list,
        raw_positions=raw_pos_list,
        positions=pos_list,
        trades=trades_list,
        oos=oos_response,
        monte_carlo=mc_response,
        deflated_sharpe=ds_response,
        regimes=regimes_response,
    )

# -------- /backtest/walk_forward --------

@router.post("/backtest/walk_forward", response_model=WalkForwardResponse)
def run_walk_forward_route(req: WalkForwardRequest) -> WalkForwardResponse:
    # 1. Look up + load + revalidate the strategy.
    with connect() as conn:
        row = repo.get_strategy(conn, req.strategy_id)
    if row is None or row["active"] == 0:
        raise HTTPException(status_code=404, detail="strategy not found")
    code_path = Path(row["code_path"])
    if not code_path.exists():
        raise HTTPException(status_code=410, detail="strategy code file missing on disk")
    code = code_path.read_text(encoding="utf-8")
    try:
        fn = load_validated_strategy(code)
    except ValueError as e:
        raise HTTPException(
            status_code=409,
            detail={"message": "stored strategy fails revalidation", "errors": [str(e)]},
        )

    # 2. Optionally wrap the strategy so base_params are merged into every grid combo.
    base = dict(req.base_params or {})
    if base:
        original_fn = fn

        def fn_with_base(data: pd.DataFrame, params: dict) -> pd.Series:
            merged = {**base, **(params or {})}
            return original_fn(data, merged)

        fn = fn_with_base

    # 3. Fetch prices.
    try:
        data = get_prices(
            ticker=req.ticker, timeframe=req.timeframe,
            start=_parse_date(req.start), end=_parse_date(req.end),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e!s}")
    if data.empty:
        raise HTTPException(status_code=404, detail="no data for requested window")

    # 4. Run the walk-forward.
    cost_config = _cost_config_from_model(req.cost_config)
    try:
        wf = walk_forward(
            fn, data, req.param_grid,
            train_bars=req.train_bars,
            test_bars=req.test_bars,
            cost_config=cost_config,
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
            selection_metric=req.selection_metric,
            higher_is_better=req.higher_is_better,
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"walk_forward failed: {e!s}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"walk_forward error: {e!s}")

    # 5. Marshal response.
    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    oos_timestamps = _index_to_iso(wf.oos_equity.index)
    oos_equity = _series_to_list(wf.oos_equity)
    oos_drawdown = _series_to_list(wf.oos_drawdown)
    oos_returns = _series_to_list(wf.oos_returns)
    oos_costs = _series_to_list(wf.oos_costs)
    oos_positions = _series_to_list(wf.oos_positions)
    oos_raw_positions = _series_to_list(wf.oos_raw_positions)
    oos_metrics_dict = _metrics_to_dict(wf.oos_metrics)
    windows_models = [_walk_forward_window_to_model(w) for w in wf.windows]
    stability_models = _param_stability_to_models(wf.param_stability)

    # 6. Persist (mode tag distinguishes this from a single backtest run).
    grid_for_persistence: Any
    if isinstance(req.param_grid, list):
        grid_for_persistence = list(req.param_grid)
    else:
        grid_for_persistence = dict(req.param_grid)
    meta_blob = {
        "mode": "walk_forward",
        "selection_metric": req.selection_metric,
        "higher_is_better": req.higher_is_better,
        "train_bars": req.train_bars,
        "test_bars": req.test_bars,
        "base_params": base,
        "param_grid": grid_for_persistence,
        "n_combinations": wf.n_combinations,
        "windows": [
            {
                "window_index": w.window_index,
                "train_start": w.train_start.isoformat(),
                "train_end": w.train_end.isoformat(),
                "test_start": w.test_start.isoformat(),
                "test_end": w.test_end.isoformat(),
                "selected_params": dict(w.selected_params),
            }
            for w in wf.windows
        ],
    }
    with connect() as conn:
        repo.insert_run(
            conn,
            run_id=run_id,
            strategy_id=req.strategy_id,
            ticker=req.ticker,
            timeframe=req.timeframe,
            start_date=req.start,
            end_date=req.end,
            params_json=json.dumps(meta_blob),
            cost_config_json=json.dumps(dataclasses.asdict(cost_config)),
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
            created_at=created_at,
        )
        repo.insert_metrics(conn, run_id=run_id, metrics_json=json.dumps(oos_metrics_dict))
        repo.insert_curves(
            conn,
            run_id=run_id,
            timestamps_json=json.dumps(oos_timestamps),
            equity_json=json.dumps(oos_equity),
            drawdown_json=json.dumps(oos_drawdown),
            returns_json=json.dumps(oos_returns),
            costs_json=json.dumps(oos_costs),
            raw_positions_json=json.dumps(oos_raw_positions),
            positions_json=json.dumps(oos_positions),
        )
        # Trades are deliberately NOT persisted for walk-forward runs:
        # per-window trade pnl_returns are window-relative, not aggregate-relative,
        # so they would be misleading on the run-level view. Phase 8 can add a
        # walk_forward_trades table if needed.

    return WalkForwardResponse(
        run_id=run_id,
        n_windows=len(wf.windows),
        n_combinations=wf.n_combinations,
        selection_metric=wf.selection_metric,
        higher_is_better=wf.higher_is_better,
        windows=windows_models,
        oos_timestamps=oos_timestamps,
        oos_equity=oos_equity,
        oos_drawdown=oos_drawdown,
        oos_returns=oos_returns,
        oos_metrics=oos_metrics_dict,
        param_stability=stability_models,
    )

