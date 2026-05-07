"""
HTTP routes. Thin layer over the engine + repo.
"""
from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import date, datetime, timezone
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
    PricesResponse,
    StrategyCreateRequest,
    StrategyCreateResponse,
    StrategyDetail,
    StrategySummary,
    ValidationResultModel,
)
from backend.data.prices import get_prices
from backend.db import repo
from backend.db.conn import connect
from backend.engine.costs import CostConfig, DEFAULT_COST_CONFIG
from backend.engine.metrics import compute_metrics
from backend.engine.validator import load_validated_strategy, validate_strategy
from backend.engine.vectorized import run_backtest
from backend.storage import STRATEGIES_DIR

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
    )
