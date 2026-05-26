"""
HTTP routes. Thin layer over the engine + repo.
"""
from __future__ import annotations

import dataclasses
import json
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.api import VERSION
from backend.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ComboResultModel,
    CostConfigModel,
    DeflatedSharpeResponseModel,
    DeploymentArmRequest,
    DeploymentCreateRequest,
    DeploymentListResponse,
    DeploymentModel,
    DeploymentRunFeedEntry,
    DeploymentRunFeedResponse,
    DeploymentRunHistoryEntry,
    DeploymentRunResponse,
    DeploymentTrackingResponse,
    RunAnnotationRequest,
    RunAnnotationResponse,
    StrategyUpdateRequest,
    StrategyVersionDetail,
    StrategyVersionEntry,
    StrategyVersionsResponse,
    EventDrivenBacktestRequest,
    FavouriteRequest,
    StopConfigModel,
    HeatmapRequestModel,
    HeatmapResponseModel,
    HealthResponse,
    KillOrderModel,
    KillSwitchRequest,
    KillSwitchResponse,
    MonteCarloConfigModel,
    MonteCarloResponseModel,
    MultiAssetResultModel,
    OOSResponseModel,
    OrderSentModel,
    ParamStabilityEntry,
    PricesResponse,
    RegimeAnalysisResponseModel,
    RegimeAxisResponseModel,
    RegimeWindowModel,
    LibraryResponse,
    LibraryRunModel,
    StrategyCreateRequest,
    StrategyCreateResponse,
    StrategyDetail,
    StrategySummary,
    SubmitResultModel,
    SweepRequest,
    SweepResponse,
    TranslateStrategyRequest,
    TranslateStrategyResponse,
    ValidateStrategyRequest,
    ValidationResultModel,
    WalkForwardRequest,
    WalkForwardResponse,
    WalkForwardWindowModel,
    WindowMetricsModel,
)
from backend.data.prices import get_prices
from backend.db import repo
from backend.db.conn import connect
from backend.deploy import scheduler
from backend.deploy.orders import cancel_working_orders
from backend.engine.costs import CostConfig, DEFAULT_COST_CONFIG
from backend.engine.event_driven import StopConfig, run_event_driven
from backend.engine.metrics import compute_metrics
from backend.engine.ranking import compute_run_scores
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
from backend.engine.robustness.heatmaps import HeatmapResult
from backend.engine.robustness.sweep import ComboResult, HeatmapRequest, SweepResult, run_sweep
from backend.engine.robustness.walk_forward import WindowReport, walk_forward
from backend.engine.validator import load_validated_strategy, validate_strategy
from backend.engine.vectorized import run_backtest
from backend.llm.groq_client import translate_strategy
from backend.storage import STRATEGIES_DIR
from backend.t212 import client as t212_client
from backend.t212 import instruments as t212_instruments

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


# ---- Phase 15.6: best-combo robustness extras for the Optimise flow ----

def _regimes_for_result(result: Any, start: date, end: date) -> RegimeAnalysisResponseModel:
    """Regime conditional analysis (SPY trend + VIX level) for a backtest
    result. Per-axis fetch errors are isolated and surfaced, never raised."""
    spy_close, spy_err = _fetch_regime_close("SPY", start, end, warmup_days=400)
    vix_close, vix_err = _fetch_regime_close("^VIX", start, end, warmup_days=0)
    regime_result = analyze_regimes(result, spy_close=spy_close, vix_close=vix_close)
    return RegimeAnalysisResponseModel(
        spy_trend=_axis_to_response(regime_result.spy_trend) if regime_result.spy_trend else None,
        vix_level=_axis_to_response(regime_result.vix_level) if regime_result.vix_level else None,
        spy_trend_error=spy_err or regime_result.spy_trend_error,
        vix_level_error=vix_err or regime_result.vix_level_error,
    )


def _retest_on_ticker(
    fn: Any, ticker: str, req: Any, cost_config: Any, params: dict,
) -> MultiAssetResultModel:
    """Re-run the best-combo params on another asset. Errors are captured
    per ticker — one bad symbol never fails the whole sweep."""
    try:
        data = get_prices(
            ticker=ticker, timeframe=req.timeframe,
            start=_parse_date(req.start), end=_parse_date(req.end),
        )
        if data is None or data.empty:
            return MultiAssetResultModel(ticker=ticker, error="no data for requested window")
        result = run_backtest(
            fn, data, params,
            cost_config=cost_config,
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
        )
        return MultiAssetResultModel(
            ticker=ticker, metrics=_metrics_to_dict(compute_metrics(result)),
        )
    except Exception as e:  # noqa: BLE001 — best-effort; isolate per ticker
        return MultiAssetResultModel(ticker=ticker, error=f"{type(e).__name__}: {e!s}")


router = APIRouter()


# -------- helpers --------

def _series_to_list(s: pd.Series) -> list[float | None]:
    """Convert a pd.Series to a JSON-safe list, mapping NaN -> None."""
    return [None if pd.isna(v) else float(v) for v in s.to_list()]


def _index_to_iso(idx: pd.DatetimeIndex) -> list[str]:
    """ISO 8601 UTC strings for a tz-aware DatetimeIndex."""
    return [ts.isoformat() for ts in idx]


def _buy_hold_equity(close: pd.Series, starting_capital: float) -> list[float | None]:
    """Buy-and-hold equity curve: the full capital invested at the first
    bar and held untouched to the end — the honest 'did the strategy beat
    doing nothing' baseline overlaid on every equity chart.

    `close` must already be aligned to the curve's timestamps; the returned
    list matches its length one-for-one (NaN -> None) so the frontend can
    overlay it bar-for-bar.
    """
    valid = close.dropna()
    if valid.empty or float(valid.iloc[0]) == 0.0:
        return [None] * len(close)
    base = float(valid.iloc[0])
    return _series_to_list(starting_capital * close / base)


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
        # Phase 16.2 — record the initial version.
        code_hash, _ = repo.insert_strategy_version(
            conn, strategy_id=strategy_id, code=req.code, created_at=created_at,
        )

    return StrategyCreateResponse(
        strategy_id=strategy_id,
        validation_result=_validation_result_to_model(vresult),
        code_hash=code_hash,
    )


@router.put("/strategies/{strategy_id}", response_model=StrategyCreateResponse)
def update_strategy_route(
    strategy_id: str, req: StrategyUpdateRequest,
) -> StrategyCreateResponse:
    """Phase 16.2 — replace a strategy's name, description, and code.
    Inserts a new strategy_versions row when the code hash is new;
    identical code is a no-op for the versions table but still updates
    the display metadata."""
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

    with connect() as conn:
        row = repo.get_strategy(conn, strategy_id)
    if row is None or row["active"] == 0:
        raise HTTPException(status_code=404, detail="strategy not found")

    # Write the new code to disk first; if anything below fails, the
    # earlier version row in strategy_versions is the safety net.
    code_path = Path(row["code_path"])
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(req.code, encoding="utf-8")

    updated_at = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        repo.update_strategy_meta(
            conn,
            strategy_id=strategy_id,
            name=req.name,
            description=req.description,
        )
        code_hash, _ = repo.insert_strategy_version(
            conn, strategy_id=strategy_id, code=req.code, created_at=updated_at,
        )

    return StrategyCreateResponse(
        strategy_id=strategy_id,
        validation_result=_validation_result_to_model(vresult),
        code_hash=code_hash,
    )


@router.post("/strategies/validate", response_model=ValidationResultModel)
def validate_strategy_route(req: ValidateStrategyRequest) -> ValidationResultModel:
    """Dry-run validation. No DB write. Always returns 200 — `ok` says it all."""
    return _validation_result_to_model(validate_strategy(req.code))


@router.post("/strategies/translate", response_model=TranslateStrategyResponse)
def translate_strategy_route(req: TranslateStrategyRequest) -> TranslateStrategyResponse:
    """Translate plain English -> Python via Groq. Returns code AND its
    pre-validation result so the UI can warn before saving."""
    try:
        code = translate_strategy(req.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM translation failed: {e!s}")

    vresult = validate_strategy(code)
    return TranslateStrategyResponse(
        code=code,
        validation_result=_validation_result_to_model(vresult),
    )


@router.get("/strategies", response_model=list[StrategySummary])
def list_strategies() -> list[StrategySummary]:
    with connect() as conn:
        rows = repo.list_strategies(conn, include_inactive=False)
    return [
        StrategySummary(
            id=r["id"], name=r["name"], description=r["description"],
            created_at=r["created_at"], favourite=bool(r["favourite"]),
        )
        for r in rows
    ]


@router.get("/strategies/{strategy_id}", response_model=StrategyDetail)
def get_strategy(strategy_id: str) -> StrategyDetail:
    with connect() as conn:
        row = repo.get_strategy(conn, strategy_id)
        current_hash = repo.get_latest_strategy_version_hash(conn, strategy_id) if row else None
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
        favourite=bool(row["favourite"]),
        code=code,
        code_hash=current_hash,
    )


# -------- /strategies/{id}/versions (Phase 16.2) --------

@router.get("/strategies/{strategy_id}/versions", response_model=StrategyVersionsResponse)
def list_strategy_versions_route(strategy_id: str) -> StrategyVersionsResponse:
    """Most-recent-first version history of a strategy's code."""
    with connect() as conn:
        row = repo.get_strategy(conn, strategy_id)
        if row is None or row["active"] == 0:
            raise HTTPException(status_code=404, detail="strategy not found")
        rows = repo.list_strategy_versions(conn, strategy_id)
    current_hash = rows[0]["code_hash"] if rows else None
    return StrategyVersionsResponse(
        strategy_id=strategy_id,
        current_hash=current_hash,
        versions=[
            StrategyVersionEntry(
                id=r["id"],
                code_hash=r["code_hash"],
                created_at=r["created_at"],
                is_current=(r["code_hash"] == current_hash),
            )
            for r in rows
        ],
    )


@router.get(
    "/strategies/{strategy_id}/versions/{code_hash}",
    response_model=StrategyVersionDetail,
)
def get_strategy_version_route(
    strategy_id: str, code_hash: str,
) -> StrategyVersionDetail:
    """Fetch the code body of a single historical version."""
    with connect() as conn:
        strat = repo.get_strategy(conn, strategy_id)
        if strat is None or strat["active"] == 0:
            raise HTTPException(status_code=404, detail="strategy not found")
        row = repo.get_strategy_version_code(conn, strategy_id, code_hash)
        if row is None:
            raise HTTPException(status_code=404, detail="version not found")
        current_hash = repo.get_latest_strategy_version_hash(conn, strategy_id)
    return StrategyVersionDetail(
        strategy_id=strategy_id,
        code_hash=row["code_hash"],
        code=row["code"],
        created_at=row["created_at"],
        is_current=(row["code_hash"] == current_hash),
    )


@router.post("/strategies/{strategy_id}/favourite", response_model=StrategySummary)
def set_strategy_favourite(strategy_id: str, req: FavouriteRequest) -> StrategySummary:
    """Star or un-star a strategy. Favourites pin to the top of the
    sidebar and the library. 404 if the strategy is unknown or deleted."""
    with connect() as conn:
        ok = repo.set_strategy_favourite(conn, strategy_id, req.favourite)
        row = repo.get_strategy(conn, strategy_id) if ok else None
    if not ok or row is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    return StrategySummary(
        id=row["id"], name=row["name"], description=row["description"],
        created_at=row["created_at"], favourite=bool(row["favourite"]),
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

    # Phase 16.1 — serialize the analyses we computed so they reach the
    # DB. None for analyses the request didn't ask for; the library
    # ranking falls back gracefully on null.
    oos_payload = oos_response.model_dump() if oos_response else None
    dsr_payload = ds_response.model_dump() if ds_response else None
    mc_payload = mc_response.model_dump() if mc_response else None

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
            oos_json=json.dumps(oos_payload) if oos_payload is not None else None,
            dsr_json=json.dumps(dsr_payload) if dsr_payload is not None else None,
            monte_carlo_json=json.dumps(mc_payload) if mc_payload is not None else None,
            strategy_code_hash=repo.hash_strategy_code(code),
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

    buy_hold_list = _buy_hold_equity(
        data["close"].reindex(result.equity.index), req.starting_capital,
    )

    return BacktestResponse(
        run_id=run_id,
        strategy_id=req.strategy_id,
        metrics=metrics_dict,
        summary=summary,
        timestamps=timestamps,
        equity=equity_list,
        buy_hold_equity=buy_hold_list,
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
        strategy_code_hash=repo.hash_strategy_code(code),
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
            strategy_code_hash=repo.hash_strategy_code(code),
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


# -------- /backtest/sweep --------

def _combo_to_model(c: ComboResult) -> ComboResultModel:
    score = c.score
    score_safe = None if (score is None or (isinstance(score, float) and score != score)) else score
    return ComboResultModel(
        index=c.index,
        params=dict(c.params),
        metrics=_metrics_to_dict(c.metrics),
        score=score_safe,
        error=c.error,
    )


def _heatmap_to_model(h: HeatmapResult) -> HeatmapResponseModel:
    grid_safe = [
        [None if (isinstance(v, float) and v != v) else v for v in row]
        for row in h.grid
    ]
    return HeatmapResponseModel(
        x_param=h.x_param,
        y_param=h.y_param,
        metric_name=h.metric_name,
        x_values=list(h.x_values),
        y_values=list(h.y_values),
        grid=grid_safe,
        missing_mask=list(list(row) for row in h.missing_mask),
        agg=h.agg,
        n_cells_filled=h.n_cells_filled,
        n_cells_total=h.n_cells_total,
    )


@router.post("/backtest/sweep", response_model=SweepResponse)
def run_sweep_route(req: SweepRequest) -> SweepResponse:
    # 1. Load + revalidate strategy.
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

    # 2. Fetch prices.
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

    # 3. Run the sweep.
    cost_config = _cost_config_from_model(req.cost_config)
    hm_requests = [
        HeatmapRequest(x_param=h.x_param, y_param=h.y_param, metric_name=h.metric_name, agg=h.agg)
        for h in req.heatmaps
    ]
    try:
        sweep = run_sweep(
            fn, data, req.param_grid,
            base_params=dict(req.base_params),
            cost_config=cost_config,
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
            selection_metric=req.selection_metric,
            higher_is_better=req.higher_is_better,
            heatmap_requests=hm_requests or None,
            include_deflated_sharpe=req.include_deflated_sharpe,
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"sweep failed: {e!s}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"sweep error: {e!s}")

    # 4. Persist run.
    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    grid_for_persistence: Any
    if isinstance(req.param_grid, list):
        grid_for_persistence = list(req.param_grid)
    else:
        grid_for_persistence = dict(req.param_grid)
    meta_blob = {
        "mode": "sweep",
        "selection_metric": req.selection_metric,
        "higher_is_better": req.higher_is_better,
        "n_combinations": sweep.n_combinations,
        "best_index": sweep.best_index,
        "param_grid": grid_for_persistence,
        "base_params": dict(req.base_params),
    }
    best_metrics: dict[str, Any] = {}
    if sweep.best_combo is not None:
        best_metrics = _metrics_to_dict(sweep.best_combo.metrics)
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
            strategy_code_hash=repo.hash_strategy_code(code),
        )
        repo.insert_metrics(conn, run_id=run_id, metrics_json=json.dumps(best_metrics))

    # 4.5 Phase 15.6 — Optimise = the full robustness picture. On the best
    # combo only (per-combo would 5x runtime): a regime check and a re-test
    # of the winning params on other assets. Both are best-effort extras —
    # a failure here never fails the sweep.
    regimes_response: RegimeAnalysisResponseModel | None = None
    multi_asset_results: list[MultiAssetResultModel] = []
    best = sweep.best_combo
    if best is not None and best.full_result is not None:
        if req.include_regimes:
            try:
                regimes_response = _regimes_for_result(
                    best.full_result, _parse_date(req.start), _parse_date(req.end),
                )
            except Exception:  # noqa: BLE001 — regime check is a best-effort extra
                regimes_response = None
        seen = {req.ticker.strip().upper()}
        for tk in req.multi_asset_tickers:
            norm = tk.strip().upper()
            if not norm or norm in seen:
                continue  # skip blanks and the asset already swept
            seen.add(norm)
            multi_asset_results.append(
                _retest_on_ticker(fn, norm, req, cost_config, dict(best.params))
            )

    # 5. Marshal and return.
    ds_model: DeflatedSharpeResponseModel | None = None
    if sweep.deflated_sharpe is not None:
        ds = sweep.deflated_sharpe
        ds_model = DeflatedSharpeResponseModel(
            n_returns=ds.n_returns,
            n_trials=ds.n_trials,
            bars_per_year=ds.bars_per_year,
            sharpe_annualized=ds.sharpe_annualized,
            sharpe_per_bar=ds.sharpe_per_bar,
            skew=ds.skew,
            excess_kurtosis=ds.excess_kurtosis,
            expected_max_sharpe_annualized=ds.expected_max_sharpe_annualized,
            psr=ds.psr,
            dsr=ds.dsr,
        )

    best_timestamps = None
    best_equity_list = None
    if sweep.best_combo and sweep.best_combo.full_result:
        fr = sweep.best_combo.full_result
        best_timestamps = _index_to_iso(fr.equity.index)
        best_equity_list = _series_to_list(fr.equity)

    return SweepResponse(
        run_id=run_id,
        n_combinations=sweep.n_combinations,
        selection_metric=sweep.selection_metric,
        higher_is_better=sweep.higher_is_better,
        best_index=sweep.best_index,
        combos=[_combo_to_model(c) for c in sweep.combos],
        deflated_sharpe=ds_model,
        heatmaps=[_heatmap_to_model(h) for h in sweep.heatmaps],
        param_keys=sweep.param_keys,
        best_combo_timestamps=best_timestamps,
        best_combo_equity=best_equity_list,
        regimes=regimes_response,
        multi_asset=multi_asset_results,
    )


# -------- /library + /runs/{run_id} (Phase 8) --------

@router.get("/library", response_model=LibraryResponse)
def library_route(
    strategy_id: str | None = Query(None),
    ticker: str | None = Query(None),
) -> LibraryResponse:
    """All stored runs joined with metrics + per-run component scores.

    Filters by `strategy_id` and `ticker` (both optional, AND-ed).
    Ordering is client-side — the frontend combines component scores
    with user-chosen weights to produce a composite ranking.

    Runs are de-duplicated by config so re-running the same backtest
    doesn't pile up identical rows — only the latest run per config is
    shown. History is kept in the DB; the de-dup is display-only.
    """
    with connect() as conn:
        rows = repo.list_runs_with_metrics(
            conn, strategy_id=strategy_id, ticker=ticker,
            latest_per_config=True,
        )
        breadth_map = repo.count_distinct_tickers_per_strategy(conn)

    scored = compute_run_scores(rows, breadth_map)
    return LibraryResponse(runs=[
        LibraryRunModel(
            run_id=s.run_id,
            strategy_id=s.strategy_id,
            strategy_name=s.strategy_name,
            ticker=s.ticker,
            timeframe=s.timeframe,
            start_date=s.start_date,
            end_date=s.end_date,
            mode=s.mode,
            created_at=s.created_at,
            favourite=s.favourite,
            metrics=s.metrics,
            sharpe_score=s.sharpe_score,
            calmar_score=s.calmar_score,
            consistency_score=s.consistency_score,
            dsr_score=s.dsr_score,
            breadth_score=s.breadth_score,
            recency_score=s.recency_score,
            sharpe_source=s.sharpe_source,
            dsr=s.dsr,
            oos_sharpe=s.oos_sharpe,
            strategy_code_hash=s.strategy_code_hash,
            notes=s.notes,
            tags=s.tags,
        )
        for s in scored
    ])


# -------- /backtest/event_driven (Phase 9) --------

@router.post("/backtest/event_driven", response_model=BacktestResponse)
def run_event_driven_route(req: EventDrivenBacktestRequest) -> BacktestResponse:
    """Event-driven backtest with protective stops.

    Same response shape as POST /backtest, but the engine walks bar-by-bar
    and may close positions mid-bar when stop_loss / take_profit /
    trailing_stop are configured. Optional robustness analyses (OOS, MC,
    DSR, regimes) are not wired in this route — they're available on the
    vectorized /backtest path.
    """
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

    cost_config = _cost_config_from_model(req.cost_config)
    try:
        stop_config = StopConfig(
            stop_loss_pct=req.stop_config.stop_loss_pct,
            take_profit_pct=req.stop_config.take_profit_pct,
            trailing_stop_pct=req.stop_config.trailing_stop_pct,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid stop_config: {e!s}")

    try:
        result = run_event_driven(
            fn, data, req.params,
            stop_config=stop_config,
            cost_config=cost_config,
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"backtest failed: {e!s}")

    metrics_bundle = compute_metrics(result)
    metrics_dict = _metrics_to_dict(metrics_bundle)

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

    # Tag the persisted run so the library can distinguish event-driven.
    meta_blob = {
        "mode": "event_driven",
        "stop_config": req.stop_config.model_dump(),
        "params": req.params,
    }
    with connect() as conn:
        repo.insert_run(
            conn, run_id=run_id, strategy_id=req.strategy_id,
            ticker=req.ticker, timeframe=req.timeframe,
            start_date=req.start, end_date=req.end,
            params_json=json.dumps(meta_blob),
            cost_config_json=json.dumps(dataclasses.asdict(cost_config)),
            starting_capital=req.starting_capital,
            bars_per_year=req.bars_per_year,
            created_at=created_at,
            strategy_code_hash=repo.hash_strategy_code(code),
        )
        repo.insert_metrics(conn, run_id=run_id, metrics_json=json.dumps(metrics_dict))
        repo.insert_curves(
            conn, run_id=run_id,
            timestamps_json=json.dumps(timestamps),
            equity_json=json.dumps(equity_list),
            drawdown_json=json.dumps(drawdown_list),
            returns_json=json.dumps(returns_list),
            costs_json=json.dumps(costs_list),
            raw_positions_json=json.dumps(raw_pos_list),
            positions_json=json.dumps(pos_list),
        )
        repo.insert_trades(conn, run_id=run_id, trades=trades_list)

    summary = {
        "cagr": metrics_dict.get("cagr"),
        "sharpe": metrics_dict.get("sharpe"),
        "max_drawdown": metrics_dict.get("max_drawdown"),
        "n_trades": len(trades_list),
        "win_rate": metrics_dict.get("win_rate"),
        "final_equity": equity_list[-1] if equity_list else None,
    }

    buy_hold_list = _buy_hold_equity(
        data["close"].reindex(result.equity.index), req.starting_capital,
    )

    return BacktestResponse(
        run_id=run_id,
        strategy_id=req.strategy_id,
        metrics=metrics_dict,
        summary=summary,
        timestamps=timestamps,
        equity=equity_list,
        buy_hold_equity=buy_hold_list,
        drawdown=drawdown_list,
        returns=returns_list,
        costs=costs_list,
        raw_positions=raw_pos_list,
        positions=pos_list,
        trades=trades_list,
        oos=None,
        monte_carlo=None,
        deflated_sharpe=None,
        regimes=None,
        strategy_code_hash=repo.hash_strategy_code(code),
    )


# -------- /t212/* read-only (Phase 10) --------

# Tiny in-memory TTL cache for /t212/* routes (Phase 10.5).
# T212's v0 API enforces aggressive per-endpoint rate limits (account/cash
# 1/2s, portfolio/orders 1/5s, pies 1/30s). The browser's refresh button
# plus lazy-load on tab activation can easily double-fire within those
# windows. This cache absorbs back-to-back calls so we don't surface T212's
# 429 as a 502 for every-other-click.
#
# Per-endpoint TTLs sit comfortably above T212's documented limits so a
# genuine refresh always either serves cached data (fast) or fetches a
# fresh value (no 429). Force-refresh isn't exposed in Phase 10 — the
# Live view's refresh button trusts the cache. /t212/instruments has its
# own 24h Parquet cache and bypasses this layer.
_t212_cache: dict[str, tuple[float, Any]] = {}

# Conservative; T212 doesn't publish rates and they may tighten. Easy to tune.
T212_CACHE_TTL_SECONDS = {
    "account": 10.0,
    "positions": 15.0,
    "orders": 15.0,
    "pies": 60.0,
}


def _now() -> float:
    """Indirection for tests; returns monotonic seconds."""
    return time.monotonic()


def _t212_cached(key: str, ttl: float, fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Serve `key` from cache if its entry is < ttl seconds old, else call
    fn and store the result. Exceptions from fn are NEVER cached — a
    failed refresh leaves the previous cache state in place (so the next
    caller retries fn) and propagates the error to the route handler.
    """
    cached = _t212_cache.get(key)
    if cached is not None:
        ts, value = cached
        if _now() - ts < ttl:
            return value
    value = fn(*args, **kwargs)
    _t212_cache[key] = (_now(), value)
    return value


def _handle_t212_call(fn, *args, **kwargs):
    """Run a T212 client call and translate its exception hierarchy into
    the right HTTPException. Each route is one line: return _handle_t212_call(fn).

    Defense-in-depth 503: backend.config raises hard at import if
    T212_DEMO_KEY/T212_DEMO_SECRET are empty, so in practice this branch
    is unreachable on a running server. Kept anyway in case someone
    monkeypatches the settings or relaxes the from_env() check.
    """
    from backend.config import settings as _settings
    if not _settings.t212_demo_key or not _settings.t212_demo_secret:
        raise HTTPException(
            status_code=503,
            detail="T212 credentials not configured (T212_DEMO_KEY/T212_DEMO_SECRET).",
        )
    try:
        return fn(*args, **kwargs)
    except t212_client.T212AuthError as e:
        raise HTTPException(status_code=502, detail=f"T212 auth failed: {e!s}")
    except t212_client.T212RateLimited as e:
        # T212 rate-limited us. Pass through the retry hint when present.
        detail = {"message": str(e)}
        if e.retry_after is not None:
            detail["retry_after_seconds"] = e.retry_after
        raise HTTPException(status_code=502, detail=detail)
    except t212_client.T212UpstreamError as e:
        raise HTTPException(status_code=502, detail=f"T212 upstream error: {e!s}")
    except t212_client.T212TransportError as e:
        raise HTTPException(status_code=504, detail=f"T212 unreachable: {e!s}")
    except t212_client.T212Error as e:
        raise HTTPException(status_code=502, detail=f"T212 error: {e!s}")


@router.get("/t212/account")
def t212_account_route() -> dict[str, Any]:
    """T212 Demo account cash breakdown.

    Returns the raw T212 cash dict: {free, total, ppl, result, invested,
    pieCash, blocked}. No transformation — the frontend renders it
    directly so new T212 fields surface automatically. Cached for 10s
    to absorb rapid refresh clicks within T212's per-endpoint cooldown.
    """
    return _handle_t212_call(
        _t212_cached, "account", T212_CACHE_TTL_SECONDS["account"],
        t212_client.get_account_cash,
    )


@router.get("/t212/positions")
def t212_positions_route() -> list[dict[str, Any]]:
    """T212 Demo open positions. Empty list when nothing is held. 15s cache."""
    return _handle_t212_call(
        _t212_cached, "positions", T212_CACHE_TTL_SECONDS["positions"],
        t212_client.get_portfolio,
    )


@router.get("/t212/orders")
def t212_orders_route() -> list[dict[str, Any]]:
    """T212 Demo working (open) orders. 15s cache."""
    return _handle_t212_call(
        _t212_cached, "orders", T212_CACHE_TTL_SECONDS["orders"],
        t212_client.get_orders,
    )


@router.get("/t212/instruments")
def t212_instruments_route(
    refresh: bool = Query(False, description="Bypass the 24h cache and refetch."),
) -> list[dict[str, Any]]:
    """T212 tradable instruments, served from the daily Parquet cache.

    Pass ?refresh=true to force a fetch. The full list is ~17K entries /
    several MB; the frontend should search/filter client-side rather
    than render everything at once.
    """
    return _handle_t212_call(t212_instruments.get_instruments, force_refresh=refresh)


@router.get("/t212/pies")
def t212_pies_route() -> list[dict[str, Any]]:
    """T212 Demo pies (auto-invest portfolios). 60s cache — T212's pies
    endpoint has the strictest per-minute limit of any /equity/ route."""
    return _handle_t212_call(
        _t212_cached, "pies", T212_CACHE_TTL_SECONDS["pies"],
        t212_client.get_pies,
    )


@router.patch("/runs/{run_id}", response_model=RunAnnotationResponse)
def patch_run_annotations(
    run_id: str, req: RunAnnotationRequest,
) -> RunAnnotationResponse:
    """Phase 16.3 — update a run's notes and/or tags. Omitted fields are
    left unchanged; explicit null clears. Tag values are normalized
    (lowercased, dedup'd, non-alnum stripped) before storage."""
    with connect() as conn:
        row = repo.get_run(conn, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")

        # Pydantic v2: model_fields_set tells us which fields were
        # actually present in the request body, so omission means
        # "leave alone" and an explicit null means "clear".
        provided = req.model_fields_set
        new_notes = row.get("notes")
        new_tags = row.get("tags")
        if "notes" in provided:
            new_notes = req.notes if req.notes else None
        if "tags" in provided:
            new_tags = repo.normalize_tags(req.tags)

        repo.update_run_notes_tags(
            conn, run_id=run_id, notes=new_notes, tags=new_tags,
        )

    return RunAnnotationResponse(
        run_id=run_id,
        notes=new_notes,
        tags=[t for t in (new_tags or "").split(",") if t],
    )


@router.get("/runs/{run_id}", response_model=BacktestResponse)
def get_run_route(run_id: str) -> BacktestResponse:
    """Load a stored run in the same shape as POST /backtest's response.

    Sweep / walk-forward runs have empty `trades` (deliberately not
    persisted in earlier phases — their trade pnl_returns are
    window-relative and would be misleading on a run-level view).
    """
    with connect() as conn:
        run_row = repo.get_run(conn, run_id)
        if run_row is None:
            raise HTTPException(status_code=404, detail="run not found")
        metrics = repo.get_metrics(conn, run_id) or {}
        curves = repo.get_curves(conn, run_id)
        trades = repo.get_trades(conn, run_id)

    if curves is None:
        raise HTTPException(status_code=410, detail="run curves missing on disk")

    equity_list = curves["equity"]
    summary = {
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "n_trades": len(trades),
        "win_rate": metrics.get("win_rate"),
        "final_equity": equity_list[-1] if equity_list else None,
    }

    # Buy-and-hold baseline — refetch prices (cached Parquet, fast) and align
    # to the stored timestamps. Best-effort: a run whose window no longer
    # refetches simply renders without the overlay.
    buy_hold_list: list[float | None] | None = None
    try:
        bh_data = get_prices(
            ticker=run_row["ticker"], timeframe=run_row["timeframe"],
            start=_parse_date(run_row["start_date"]),
            end=_parse_date(run_row["end_date"]),
        )
        target_index = pd.to_datetime(curves["timestamps"], utc=True)
        close_aligned = bh_data["close"].reindex(target_index)
        buy_hold_list = _buy_hold_equity(close_aligned, run_row["starting_capital"])
    except Exception:  # noqa: BLE001 — the overlay is best-effort
        buy_hold_list = None

    # Phase 16.1 — rehydrate persisted OOS / DSR / MC if they were
    # computed at the original request. Old runs still return None.
    oos_blob = run_row.get("oos_json")
    dsr_blob = run_row.get("dsr_json")
    mc_blob = run_row.get("monte_carlo_json")
    oos_model = OOSResponseModel(**json.loads(oos_blob)) if oos_blob else None
    dsr_model = DeflatedSharpeResponseModel(**json.loads(dsr_blob)) if dsr_blob else None
    mc_model = MonteCarloResponseModel(**json.loads(mc_blob)) if mc_blob else None

    return BacktestResponse(
        run_id=run_row["id"],
        strategy_id=run_row["strategy_id"],
        metrics=metrics,
        summary=summary,
        timestamps=curves["timestamps"],
        equity=equity_list,
        buy_hold_equity=buy_hold_list,
        drawdown=curves["drawdown"],
        returns=curves["returns"],
        costs=curves["costs"],
        raw_positions=curves["raw_positions"],
        positions=curves["positions"],
        trades=trades,
        oos=oos_model,
        monte_carlo=mc_model,
        deflated_sharpe=dsr_model,
        regimes=None,
        strategy_code_hash=run_row.get("strategy_code_hash"),
        notes=run_row.get("notes"),
        tags=[t for t in (run_row.get("tags") or "").split(",") if t],
    )


# -------- /deployments (Phase 11.6) --------

# Creating or re-arming a deployment enables live order placement, so both
# require this exact confirmation token. Disarming a deployment, and a
# manual run of an already-armed one, do not.
ARM_TOKEN = "ARM"


def _deployment_to_model(row: dict[str, Any]) -> DeploymentModel:
    """deployments row (sqlite) -> DeploymentModel (API)."""
    return DeploymentModel(
        id=row["id"],
        strategy_id=row["strategy_id"],
        ticker=row["ticker"],
        timeframe=row["timeframe"],
        params=json.loads(row["params_json"]) if row["params_json"] else {},
        deployment_capital=row["deployment_capital"],
        cost_config=json.loads(row["cost_config_json"]) if row["cost_config_json"] else {},
        order_type=row["order_type"],
        schedule_cron=row["schedule_cron"],
        armed=bool(row["armed"]),
        created_at=row["created_at"],
        last_run_at=row["last_run_at"],
        max_position_value=row["max_position_value"],
        max_loss_value=row["max_loss_value"],
    )


def _order_to_model(row: dict[str, Any]) -> OrderSentModel:
    """orders_sent row (sqlite) -> OrderSentModel (API)."""
    return OrderSentModel(
        id=row["id"],
        deployment_id=row["deployment_id"],
        request_id_hash=row["request_id_hash"],
        t212_order_id=row["t212_order_id"],
        as_of_timestamp=row["as_of_timestamp"],
        ticker=row["ticker"],
        side=row["side"],
        qty=row["qty"],
        order_type=row["order_type"],
        limit_price=row["limit_price"],
        status=row["status"],
        response_json=row["response_json"],
        created_at=row["created_at"],
    )


@router.post("/deployments", response_model=DeploymentModel)
def create_deployment_route(req: DeploymentCreateRequest) -> DeploymentModel:
    """Create a Trading 212 deployment.

    Requires `confirmation_token == "ARM"`. The deployment is created
    armed and its cron job is registered immediately — the token is the
    "I understand this will place real orders" gate.
    """
    if req.confirmation_token != ARM_TOKEN:
        raise HTTPException(
            status_code=403,
            detail='confirmation_token must be "ARM" to create a deployment',
        )
    if req.order_type not in ("market", "limit"):
        raise HTTPException(status_code=400, detail="order_type must be 'market' or 'limit'")
    if req.timeframe not in ("1d", "1h", "5m"):
        raise HTTPException(status_code=400, detail="timeframe must be '1d', '1h', or '5m'")
    try:
        scheduler.validate_cron(req.schedule_cron)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid schedule_cron: {e!s}")

    deployment_id = uuid.uuid4().hex
    cost_config_json = json.dumps(
        req.cost_config.model_dump() if req.cost_config is not None else {}
    )
    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        strat = repo.get_strategy(conn, req.strategy_id)
        if strat is None or strat["active"] == 0:
            raise HTTPException(status_code=404, detail="strategy not found")
        if any(d["ticker"] == req.ticker for d in repo.list_deployments(conn)):
            raise HTTPException(
                status_code=409,
                detail=f"a deployment for {req.ticker} already exists — "
                       "delete it first (one deployment per ticker)",
            )
        repo.insert_deployment(
            conn,
            deployment_id=deployment_id,
            strategy_id=req.strategy_id,
            ticker=req.ticker,
            timeframe=req.timeframe,
            params_json=json.dumps(req.params),
            deployment_capital=req.deployment_capital,
            cost_config_json=cost_config_json,
            order_type=req.order_type,
            schedule_cron=req.schedule_cron,
            armed=True,
            created_at=created_at,
            max_position_value=req.max_position_value,
            max_loss_value=req.max_loss_value,
        )
        row = repo.get_deployment(conn, deployment_id)

    # Created armed -> register its cron job now.
    scheduler.register_deployment(row)
    return _deployment_to_model(row)


@router.get("/deployments", response_model=DeploymentListResponse)
def list_deployments_route(
    strategy_id: str | None = Query(None),
    armed_only: bool = Query(False),
) -> DeploymentListResponse:
    """List deployments, newest first. Filters (strategy_id, armed_only) are AND-ed."""
    with connect() as conn:
        rows = repo.list_deployments(conn, strategy_id=strategy_id, armed_only=armed_only)
    return DeploymentListResponse(deployments=[_deployment_to_model(r) for r in rows])


@router.get("/deployments/runs", response_model=DeploymentRunFeedResponse)
def deployment_runs_feed_route(
    since_id: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> DeploymentRunFeedResponse:
    """Phase 14 — cross-deployment fire feed for browser notifications.
    Returns rows from deployment_runs with id > since_id (newest first)
    plus the current max id, so the frontend can advance its watermark
    even when nothing fired during the poll window. Declared BEFORE
    `/deployments/{deployment_id}` so FastAPI doesn't bind `runs` to
    the path parameter."""
    with connect() as conn:
        rows = repo.list_deployment_runs_since(conn, since_id=since_id, limit=limit)
        latest_id = repo.max_deployment_run_id(conn)
    return DeploymentRunFeedResponse(
        runs=[DeploymentRunFeedEntry(**r) for r in rows],
        latest_id=latest_id,
    )


@router.get("/deployments/{deployment_id}", response_model=DeploymentModel)
def get_deployment_route(deployment_id: str) -> DeploymentModel:
    with connect() as conn:
        row = repo.get_deployment(conn, deployment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="deployment not found")
    return _deployment_to_model(row)


@router.post("/deployments/{deployment_id}/arm", response_model=DeploymentModel)
def arm_deployment_route(deployment_id: str, req: DeploymentArmRequest) -> DeploymentModel:
    """Re-arm a paused deployment. Requires the ARM token — re-enabling
    live order placement carries the same risk as creating one."""
    if req.confirmation_token != ARM_TOKEN:
        raise HTTPException(
            status_code=403,
            detail='confirmation_token must be "ARM" to arm a deployment',
        )
    with connect() as conn:
        row = repo.get_deployment(conn, deployment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        repo.set_deployment_armed(conn, deployment_id, True)
        row = repo.get_deployment(conn, deployment_id)
    scheduler.register_deployment(row)
    return _deployment_to_model(row)


@router.post("/deployments/{deployment_id}/disarm", response_model=DeploymentModel)
def disarm_deployment_route(deployment_id: str) -> DeploymentModel:
    """Pause a deployment: clears `armed` and removes its cron job. No
    token required — turning a deployment OFF is always safe and should be
    frictionless."""
    with connect() as conn:
        row = repo.get_deployment(conn, deployment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        repo.set_deployment_armed(conn, deployment_id, False)
        cancel_working_orders(conn, deployment_id=deployment_id)
        row = repo.get_deployment(conn, deployment_id)
    scheduler.unregister_deployment(deployment_id)
    return _deployment_to_model(row)


@router.post("/deployments/{deployment_id}/run", response_model=DeploymentRunResponse)
def run_deployment_route(deployment_id: str) -> DeploymentRunResponse:
    """Manually trigger one run of a deployment, now.

    Returns 200 with the outcome in `status` (OK / DISARMED / NO_DATA /
    ERROR): a disarmed or erroring run is a valid, reported outcome, not
    an HTTP error. 404 only when the deployment id is unknown.
    """
    with connect() as conn:
        row = repo.get_deployment(conn, deployment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="deployment not found")

    result = scheduler.run_deployment(deployment_id)
    return DeploymentRunResponse(
        deployment_id=result.deployment_id,
        status=result.status,
        ran_at=result.ran_at,
        as_of_timestamp=result.as_of_timestamp,
        intent_count=result.intent_count,
        submitted=sum(1 for r in result.submit_results if r.status == "SUBMITTED"),
        results=[
            SubmitResultModel(
                ticker=r.intent.ticker,
                side=r.intent.side,
                qty=r.intent.qty,
                order_type=r.intent.order_type,
                status=r.status,
                t212_order_id=r.t212_order_id,
                error=r.error,
            )
            for r in result.submit_results
        ],
        error=result.error,
    )


@router.get("/deployments/{deployment_id}/tracking", response_model=DeploymentTrackingResponse)
def deployment_tracking_route(deployment_id: str) -> DeploymentTrackingResponse:
    """Expected-vs-actual for a deployment.

    `orders` is every order the deployment has recorded. `expected_*` is a
    backtest of the deployment's strategy over the same price window the
    scheduler fetches — best-effort: if it cannot be produced the curves
    come back empty and `expected_error` says why, but the deployment and
    its orders are always returned. 404 only when the deployment is unknown.
    """
    with connect() as conn:
        row = repo.get_deployment(conn, deployment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        order_rows = repo.list_orders_for_deployment(conn, deployment_id)
        run_rows = repo.list_deployment_runs(conn, deployment_id, limit=50)
        strat = repo.get_strategy(conn, row["strategy_id"])

    expected_timestamps: list[str] = []
    expected_equity: list[float | None] = []
    expected_buy_hold: list[float | None] = []
    expected_positions: list[float | None] = []
    expected_metrics: dict[str, Any] = {}
    expected_error: str | None = None
    expected_rebased: bool = False
    try:
        if strat is None or strat["active"] == 0:
            raise RuntimeError("strategy not found or inactive")
        code_path = Path(strat["code_path"])
        if not code_path.exists():
            raise RuntimeError("strategy code file missing on disk")
        fn = load_validated_strategy(code_path.read_text(encoding="utf-8"))

        yf_ticker = scheduler.resolve_price_ticker(row["ticker"])
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=scheduler.lookback_days(row["timeframe"]))
        data = get_prices(yf_ticker, row["timeframe"], start, end)
        if data is None or data.empty:
            raise RuntimeError("no price data for the deployment window")

        cost_json = json.loads(row["cost_config_json"]) if row["cost_config_json"] else {}
        cost_config = _cost_config_from_model(
            CostConfigModel(**cost_json) if cost_json else None
        )
        result = run_backtest(
            fn,
            data,
            json.loads(row["params_json"]) if row["params_json"] else {},
            cost_config=cost_config,
            starting_capital=row["deployment_capital"],
            bars_per_year=252,
        )
        expected_metrics = _metrics_to_dict(compute_metrics(result))

        # Phase 14 — rebase the expected curve to the deployment's live
        # period so the visual comparison against the orders below is
        # apples-to-apples. Metrics stay over the full lookback window —
        # they describe the strategy's history, not just this slice.
        # Fall back to the full window for a brand-new deployment whose
        # created_at is later than every available bar.
        created_ts = pd.Timestamp(row["created_at"])
        live_index = result.equity.index[result.equity.index >= created_ts]
        if len(live_index) > 0:
            equity_slice = result.equity.loc[live_index]
            first_eq = float(equity_slice.iloc[0])
            scale = (row["deployment_capital"] / first_eq) if first_eq > 0 else 1.0
            equity_rebased = equity_slice * scale

            expected_timestamps = _index_to_iso(equity_rebased.index)
            expected_equity = _series_to_list(equity_rebased)
            expected_positions = _series_to_list(result.positions.loc[live_index])
            expected_buy_hold = _buy_hold_equity(
                data["close"].reindex(equity_rebased.index),
                row["deployment_capital"],
            )
            expected_rebased = True
        else:
            expected_timestamps = _index_to_iso(result.equity.index)
            expected_equity = _series_to_list(result.equity)
            expected_positions = _series_to_list(result.positions)
            expected_buy_hold = _buy_hold_equity(
                data["close"].reindex(result.equity.index),
                row["deployment_capital"],
            )
    except Exception as e:  # noqa: BLE001 — the expected curve is best-effort
        expected_error = f"{type(e).__name__}: {e!s}"

    return DeploymentTrackingResponse(
        deployment=_deployment_to_model(row),
        orders=[_order_to_model(r) for r in order_rows],
        runs=[DeploymentRunHistoryEntry(**r) for r in run_rows],
        expected_timestamps=expected_timestamps,
        expected_equity=expected_equity,
        expected_buy_hold=expected_buy_hold,
        expected_positions=expected_positions,
        expected_metrics=expected_metrics,
        expected_error=expected_error,
        expected_rebased=expected_rebased,
    )


@router.post("/killswitch", response_model=KillSwitchResponse)
def killswitch_route(req: KillSwitchRequest) -> KillSwitchResponse:
    """Kill switch — disarm every deployment and flatten every open T212
    position with market closing orders.

    Requires `confirmation_token == "KILL"`. Always returns 200 with a
    full report — an emergency action must never fail opaquely. Disarming
    happens first (local, reliable); the flatten follows and reports each
    position's outcome individually so one rejection can't hide the rest.
    """
    if req.confirmation_token != "KILL":
        raise HTTPException(
            status_code=403,
            detail='confirmation_token must be "KILL" to trigger the kill switch',
        )

    # 1. Disarm every deployment — local DB + scheduler, always succeeds.
    with connect() as conn:
        deployments = repo.list_deployments(conn)
        for d in deployments:
            repo.set_deployment_armed(conn, d["id"], False)
            cancel_working_orders(conn, deployment_id=d["id"])
    for d in deployments:
        scheduler.unregister_deployment(d["id"])

    # 2. Flatten every open position with a market closing order.
    orders: list[KillOrderModel] = []
    fetch_error: str | None = None
    try:
        portfolio = t212_client.get_portfolio()
    except t212_client.T212Error as e:
        portfolio = []
        fetch_error = f"could not fetch open positions to flatten: {e!s}"

    for pos in portfolio:
        ticker = pos.get("ticker")
        qty = float(pos.get("quantity") or 0.0)
        if not ticker or abs(qty) < 1e-9:
            continue
        try:
            resp = t212_client.place_market_order(ticker, -qty)  # close the position
            oid = resp.get("id") if isinstance(resp, dict) else None
            orders.append(KillOrderModel(
                ticker=ticker, qty=qty, status="SUBMITTED",
                t212_order_id=int(oid) if oid is not None else None, error=None,
            ))
        except t212_client.T212Error as e:
            orders.append(KillOrderModel(
                ticker=ticker, qty=qty, status="FAILED",
                t212_order_id=None, error=str(e),
            ))

    return KillSwitchResponse(
        deployments_disarmed=len(deployments),
        orders=orders,
        error=fetch_error,
    )


@router.delete("/deployments/{deployment_id}")
def delete_deployment_route(deployment_id: str) -> dict[str, Any]:
    """Delete a deployment — cancels any still-working order, then removes
    the row and (via ON DELETE CASCADE) its order + fill history. There is
    no separate edit route: a strategy/ticker change is better expressed
    as delete + recreate, and capital/limit changes likewise."""
    with connect() as conn:
        row = repo.get_deployment(conn, deployment_id)
        if row is None:
            raise HTTPException(status_code=404, detail="deployment not found")
        cancel_working_orders(conn, deployment_id=deployment_id)
        repo.delete_deployment(conn, deployment_id)
    scheduler.unregister_deployment(deployment_id)
    return {"deployment_id": deployment_id, "deleted": True}
