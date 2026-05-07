"""
Pydantic v2 request/response models for the API.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------- /health ----------
class HealthResponse(BaseModel):
    status: str
    version: str


# ---------- /prices ----------
class PricesResponse(BaseModel):
    ticker: str
    timeframe: str
    timestamps: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]


# ---------- /strategies ----------
class StrategyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    code: str = Field(min_length=1)


class ValidationResultModel(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    strategy_name: str | None = None


class StrategyCreateResponse(BaseModel):
    strategy_id: str
    validation_result: ValidationResultModel


class StrategySummary(BaseModel):
    id: str
    name: str
    description: str
    created_at: str


class StrategyDetail(StrategySummary):
    code: str


# ---------- /backtest ----------
class CostConfigModel(BaseModel):
    """Mirror of backend.engine.costs.CostConfig. Defaults match DEFAULT_COST_CONFIG."""
    model_config = ConfigDict(extra="forbid")

    slippage_mode: str = "fixed"        # "fixed" | "vol_scaled"
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    spread_bps: float = 0.0
    fx_fee_bps: float = 15.0
    instrument_is_gbp: bool = False
    borrow_cost_bps: float = 0.0


class BacktestRequest(BaseModel):
    strategy_id: str
    ticker: str
    timeframe: str = "1d"
    start: str  # ISO date "YYYY-MM-DD"
    end: str    # ISO date "YYYY-MM-DD"
    params: dict[str, Any] = Field(default_factory=dict)
    cost_config: CostConfigModel | None = None
    starting_capital: float = 100_000.0
    bars_per_year: int = 252


class TradeModel(BaseModel):
    """Loose schema — populated from dataclasses.asdict(trade). Extra fields permitted."""
    model_config = ConfigDict(extra="allow")


class BacktestResponse(BaseModel):
    run_id: str
    metrics: dict[str, Any]
    summary: dict[str, Any]            # {cagr, sharpe, max_dd, n_trades, win_rate, final_equity}
    timestamps: list[str]
    equity: list[float | None]
    drawdown: list[float | None]
    returns: list[float | None]
    costs: list[float | None]
    raw_positions: list[float | None]
    positions: list[float | None]
    trades: list[dict[str, Any]]
