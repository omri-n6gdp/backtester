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


class ValidateStrategyRequest(BaseModel):
    code: str = Field(min_length=1)


class TranslateStrategyRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)


class TranslateStrategyResponse(BaseModel):
    code: str
    validation_result: ValidationResultModel


# ---------- /library (Phase 8) ----------
class LibraryRunModel(BaseModel):
    """One ranked run in the strategy library."""
    run_id: str
    strategy_id: str
    strategy_name: str
    ticker: str
    timeframe: str
    start_date: str
    end_date: str
    mode: str                          # 'single' | 'sweep' | 'walk_forward'
    created_at: str
    metrics: dict[str, Any]
    # Per-component normalized scores in [0, 1]. dsr_score is null for now.
    sharpe_score: float
    calmar_score: float
    consistency_score: float
    dsr_score: float | None
    breadth_score: float
    recency_score: float


class LibraryResponse(BaseModel):
    runs: list[LibraryRunModel]


# ---------- /backtest ----------
class CostConfigModel(BaseModel):
    """Mirror of backend.engine.costs.CostConfig. Field names match the dataclass."""
    model_config = ConfigDict(extra="forbid")

    slippage_mode: str = "fixed"        # "fixed" | "vol_scaled"
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    spread_bps: float = 0.0
    fx_bps: float = 15.0
    instrument_is_gbp: bool = False
    borrow_bps_annual: float = 0.0
    bars_per_year: int = 252


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
    oos_split: float | str | None = None
    monte_carlo: MonteCarloConfigModel | None = None
    n_trials: int | None = None
    include_regimes: bool = False


class StopConfigModel(BaseModel):
    """Mirror of backend.engine.event_driven.StopConfig.

    All percents are non-negative fractions: 0.05 = 5%. None disables the
    corresponding stop. At least one stop must be active for the
    event-driven route to be meaningful (the validator enforces this).
    """
    model_config = ConfigDict(extra="forbid")

    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None


class EventDrivenBacktestRequest(BaseModel):
    """Same shape as BacktestRequest plus a stop_config block.

    The optional analyses (oos_split, monte_carlo, n_trials,
    include_regimes) are deliberately NOT supported here yet — they're
    computed at request time and don't depend on the execution model,
    so they could be wired in later without touching the engine.
    """
    strategy_id: str
    ticker: str
    timeframe: str = "1d"
    start: str
    end: str
    params: dict[str, Any] = Field(default_factory=dict)
    cost_config: CostConfigModel | None = None
    starting_capital: float = 100_000.0
    bars_per_year: int = 252
    stop_config: StopConfigModel = Field(default_factory=StopConfigModel)


class WindowMetricsModel(BaseModel):
    """IS or OOS window of a backtest, with the window's MetricsBundle."""
    label: str                 # 'in_sample' or 'out_of_sample'
    start: str                 # ISO timestamp of first bar
    end: str                   # ISO timestamp of last bar
    n_bars: int
    metrics: dict[str, Any]


class OOSResponseModel(BaseModel):
    """Result of an in-sample / out-of-sample split on a backtest."""
    split_timestamp: str       # first OOS bar
    split_index: int
    in_sample: WindowMetricsModel
    out_of_sample: WindowMetricsModel


class WalkForwardRequest(BaseModel):
    """Request body for POST /backtest/walk_forward."""
    strategy_id: str
    ticker: str
    timeframe: str = "1d"
    start: str
    end: str
    base_params: dict[str, Any] = Field(default_factory=dict)
    param_grid: dict[str, Any] | list[dict[str, Any]]
    train_bars: int
    test_bars: int
    cost_config: CostConfigModel | None = None
    starting_capital: float = 100_000.0
    bars_per_year: int = 252
    selection_metric: str = "sharpe"
    higher_is_better: bool = True


class WalkForwardWindowModel(BaseModel):
    """Per-window report inside a WalkForwardResponse."""
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_bars: int
    n_test_bars: int
    selected_params: dict[str, Any]
    is_metrics: dict[str, Any]
    oos_metrics: dict[str, Any]


class ParamStabilityEntry(BaseModel):
    """For a single parameter key: how stable was its selection across windows.

    `counts` keys are stringified parameter values (JSON requires string keys).
    `mode` is the stringified most-frequently-selected value.
    """
    counts: dict[str, int]
    n_distinct: int
    mode: str


class WalkForwardResponse(BaseModel):
    """Response from POST /backtest/walk_forward."""
    run_id: str
    n_windows: int
    n_combinations: int
    selection_metric: str
    higher_is_better: bool
    windows: list[WalkForwardWindowModel]
    oos_timestamps: list[str]
    oos_equity: list[float | None]
    oos_drawdown: list[float | None]
    oos_returns: list[float | None]
    oos_metrics: dict[str, Any]
    param_stability: dict[str, ParamStabilityEntry]


class MonteCarloConfigModel(BaseModel):
    """Monte Carlo trade-order shuffling config. Mirror of MonteCarloConfig."""
    model_config = ConfigDict(extra="forbid")

    n_iterations: int = 1000
    seed: int | None = 42
    quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.5, 0.95])


class MonteCarloResponseModel(BaseModel):
    """Result of a Monte Carlo trade-order shuffle on a backtest."""
    n_iterations: int
    n_trades: int
    seed: int | None
    quantiles: list[float]
    original_final_equity: float
    original_max_drawdown: float
    original_max_drawdown_trades: int
    final_equities: list[float]
    max_drawdowns: list[float]
    max_drawdown_durations: list[int]
    equity_bands: dict[str, list[float]]
    drawdown_bands: dict[str, list[float]]
    max_dd_quantiles: dict[str, float]
    original_dd_percentile: float | None


class DeflatedSharpeResponseModel(BaseModel):
    """Probabilistic + Deflated Sharpe Ratio analysis. Mirror of DeflatedSharpeResult."""
    n_returns: int
    n_trials: int
    bars_per_year: int
    sharpe_annualized: float | None
    sharpe_per_bar: float | None
    skew: float | None
    excess_kurtosis: float | None
    expected_max_sharpe_annualized: float | None
    psr: float | None
    dsr: float | None


class RegimeWindowModel(BaseModel):
    """One regime's slice of a backtest with its bar-level metrics."""
    label: str
    n_bars: int
    fraction: float
    metrics: dict[str, Any]


class RegimeAxisResponseModel(BaseModel):
    """One regime axis (e.g. SPY trend, VIX level)."""
    name: str
    n_bars_total: int
    n_bars_unknown: int
    windows: dict[str, RegimeWindowModel]


class RegimeAnalysisResponseModel(BaseModel):
    """Result of regime conditional analysis on a backtest."""
    spy_trend: RegimeAxisResponseModel | None = None
    vix_level: RegimeAxisResponseModel | None = None
    spy_trend_error: str | None = None
    vix_level_error: str | None = None


class HeatmapRequestModel(BaseModel):
    """One (x_param, y_param, metric_name) heatmap to render in a sweep."""
    x_param: str
    y_param: str
    metric_name: str
    agg: str = "last"


class HeatmapResponseModel(BaseModel):
    """2D metric grid across two parameter axes. Mirrors HeatmapResult."""
    x_param: str
    y_param: str
    metric_name: str
    x_values: list[Any]
    y_values: list[Any]
    grid: list[list[float | None]]
    missing_mask: list[list[bool]]
    agg: str
    n_cells_filled: int
    n_cells_total: int


class ComboResultModel(BaseModel):
    """Per-combo summary from a sweep."""
    index: int
    params: dict[str, Any]
    metrics: dict[str, Any]
    score: float | None
    error: str | None = None


class SweepRequest(BaseModel):
    """Request body for POST /backtest/sweep."""
    strategy_id: str
    ticker: str
    timeframe: str = "1d"
    start: str
    end: str
    base_params: dict[str, Any] = Field(default_factory=dict)
    param_grid: dict[str, Any] | list[dict[str, Any]]
    cost_config: CostConfigModel | None = None
    starting_capital: float = 100_000.0
    bars_per_year: int = 252
    selection_metric: str = "sharpe"
    higher_is_better: bool = True
    heatmaps: list[HeatmapRequestModel] = Field(default_factory=list)
    include_deflated_sharpe: bool = True


class SweepResponse(BaseModel):
    """Response from POST /backtest/sweep."""
    run_id: str
    n_combinations: int
    selection_metric: str
    higher_is_better: bool
    best_index: int | None
    combos: list[ComboResultModel]
    deflated_sharpe: DeflatedSharpeResponseModel | None = None
    heatmaps: list[HeatmapResponseModel]
    param_keys: list[str]
    best_combo_timestamps: list[str] | None = None
    best_combo_equity: list[float | None] | None = None


class TradeModel(BaseModel):
    """Loose schema — populated from dataclasses.asdict(trade). Extra fields permitted."""
    model_config = ConfigDict(extra="allow")


class BacktestResponse(BaseModel):
    run_id: str
    strategy_id: str | None = None
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
    oos: OOSResponseModel | None = None
    monte_carlo: MonteCarloResponseModel | None = None
    deflated_sharpe: DeflatedSharpeResponseModel | None = None
    regimes: RegimeAnalysisResponseModel | None = None


# ---------- /deployments (Phase 11.6) ----------
class DeploymentCreateRequest(BaseModel):
    """Create a Trading 212 deployment.

    `confirmation_token` must be the literal "ARM". A deployment is
    created armed and starts running on its cron schedule immediately, so
    the token is the "I understand this will place real orders" gate.
    """
    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)               # T212 instrument code, e.g. "AAPL_US_EQ"
    timeframe: str = "1d"
    params: dict[str, Any] = Field(default_factory=dict)
    deployment_capital: float = Field(gt=0)
    cost_config: CostConfigModel | None = None
    order_type: str = "market"                      # "market" | "limit"
    schedule_cron: str = Field(min_length=1)        # 5-field crontab, interpreted in UTC
    confirmation_token: str = Field(min_length=1)


class DeploymentArmRequest(BaseModel):
    """Body for POST /deployments/{id}/arm. Re-enabling live order
    placement requires the ARM token, exactly as creation does."""
    model_config = ConfigDict(extra="forbid")

    confirmation_token: str = Field(min_length=1)


class DeploymentModel(BaseModel):
    """A deployment row as returned by the API."""
    id: str
    strategy_id: str
    ticker: str
    timeframe: str
    params: dict[str, Any]
    deployment_capital: float
    cost_config: dict[str, Any]
    order_type: str
    schedule_cron: str
    armed: bool
    created_at: str
    last_run_at: str | None = None


class DeploymentListResponse(BaseModel):
    deployments: list[DeploymentModel]


class OrderSentModel(BaseModel):
    """One row from orders_sent — an order the deployment recorded."""
    id: str
    deployment_id: str
    request_id_hash: str
    t212_order_id: int | None = None
    as_of_timestamp: str
    ticker: str
    side: str
    qty: float
    order_type: str
    limit_price: float | None = None
    status: str
    response_json: str | None = None
    created_at: str


class SubmitResultModel(BaseModel):
    """Per-intent outcome of one deployment run."""
    ticker: str
    side: str
    qty: float
    order_type: str
    status: str                 # SUBMITTED | SKIPPED_DUPLICATE | FAILED
    t212_order_id: int | None = None
    error: str | None = None


class DeploymentRunResponse(BaseModel):
    """Result of POST /deployments/{id}/run (manual trigger)."""
    deployment_id: str
    status: str                 # OK | DISARMED | NO_DATA | ERROR
    ran_at: str
    as_of_timestamp: str | None = None
    intent_count: int
    submitted: int
    results: list[SubmitResultModel]
    error: str | None = None


class DeploymentTrackingResponse(BaseModel):
    """Expected (backtest) vs actual (orders sent) for a deployment.

    `expected_*` is a backtest of the deployment's strategy over the same
    price window the scheduler uses. If that backtest cannot be produced
    (data fetch failed, stored code invalid, ...), the curves come back
    empty and `expected_error` explains why — the deployment and its
    orders are always returned.
    """
    deployment: DeploymentModel
    orders: list[OrderSentModel]
    expected_timestamps: list[str]
    expected_equity: list[float | None]
    expected_positions: list[float | None]
    expected_metrics: dict[str, Any]
    expected_error: str | None = None
