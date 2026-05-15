"""
Composite ranking for stored backtest runs (Phase 8).

Pure functions, zero I/O. Given a list of run rows (from
`repo.list_runs_with_metrics`) plus a breadth map, produce per-run
normalized component scores in [0, 1]:

  - sharpe_score      = clamp(sharpe / 2.0, 0, 1)
  - calmar_score      = clamp(calmar / 3.0, 0, 1)
  - consistency_score = clamp(sortino / max(sharpe, ε), 0, 2) / 2
  - dsr_score         = None (DSR isn't persisted today)
  - breadth_score     = log1p(distinct_tickers - 1) / log(6) saturating ~5
  - recency_score     = exp(-days_old / 90)

No internal weighting — the frontend combines these with user weights.
DSR is left null so the frontend can redistribute its weight to
available components per row (otherwise DSR-less runs would be
unfairly zeroed).

The default weights from CLAUDE.md (30/20/15/15/10/10) live in the
frontend; the backend doesn't need them for ordering since ordering
is client-side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# Saturation knobs. These are intentional — Sharpe > 2 is rare and shouldn't
# linearly extend the score; Calmar > 3 likewise. Calibrated against typical
# robust-strategy ranges (Sharpe 1-2, Calmar 1-3 on free daily data).
_SHARPE_SAT = 2.0
_CALMAR_SAT = 3.0
_BREADTH_SAT_TICKERS = 6  # saturates around 5 tickers
_RECENCY_HALFLIFE_DAYS = 90.0


@dataclass
class ScoredRun:
    """A run row with normalized component scores. All component scores are
    in [0, 1] when set; `dsr_score` is None for now."""
    run_id: str
    strategy_id: str
    strategy_name: str
    ticker: str
    timeframe: str
    start_date: str
    end_date: str
    mode: str
    created_at: str
    metrics: dict[str, Any]
    # Raw metric snapshot (already in metrics, but copied for convenience):
    sharpe: float | None
    calmar: float | None
    cagr: float | None
    max_drawdown: float | None
    sortino: float | None
    n_trades: int | None
    # Normalized component scores:
    sharpe_score: float
    calmar_score: float
    consistency_score: float
    dsr_score: float | None
    breadth_score: float
    recency_score: float


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_float(x: Any) -> float | None:
    """Coerce to float; return None on NaN / None / coerce errors."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def _sharpe_score(sharpe: float | None) -> float:
    if sharpe is None:
        return 0.0
    return _clamp01(sharpe / _SHARPE_SAT)


def _calmar_score(calmar: float | None) -> float:
    if calmar is None:
        return 0.0
    return _clamp01(calmar / _CALMAR_SAT)


def _consistency_score(sharpe: float | None, sortino: float | None) -> float:
    """Sortino/Sharpe ratio scaled to [0, 1]. Sortino >= Sharpe when downside
    volatility < total volatility, so the ratio is typically in [1, ~2]. We
    clamp at 2 and divide.

    When sharpe is non-positive or sortino is missing, we have no signal —
    return 0 rather than fabricate one.
    """
    if sharpe is None or sortino is None or sharpe <= 0:
        return 0.0
    ratio = sortino / sharpe
    return _clamp01(ratio / 2.0)


def _breadth_score(distinct_tickers: int) -> float:
    """Log saturation so 1 ticker scores 0, 2 ~0.39, 5 ~0.92, 6+ = 1."""
    if distinct_tickers <= 1:
        return 0.0
    return _clamp01(math.log(distinct_tickers) / math.log(_BREADTH_SAT_TICKERS))


def _recency_score(created_at: str, now: datetime) -> float:
    """exp(-days_old / 90). 0 days = 1.0; 90 days = 0.37; 180 days = 0.14."""
    try:
        ts = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days_old = max((now - ts).total_seconds() / 86400.0, 0.0)
    return math.exp(-days_old / _RECENCY_HALFLIFE_DAYS)


def compute_run_scores(
    rows: list[dict[str, Any]],
    breadth_map: dict[str, int],
    *,
    now: datetime | None = None,
) -> list[ScoredRun]:
    """Compute per-run component scores. Order matches input order."""
    now = now or datetime.now(timezone.utc)
    out: list[ScoredRun] = []
    for r in rows:
        m = r.get("metrics") or {}
        sharpe = _safe_float(m.get("sharpe"))
        calmar = _safe_float(m.get("calmar"))
        sortino = _safe_float(m.get("sortino"))
        cagr = _safe_float(m.get("cagr"))
        max_dd = _safe_float(m.get("max_drawdown"))
        n_trades_raw = m.get("n_trades")
        n_trades = int(n_trades_raw) if n_trades_raw is not None else None

        distinct = breadth_map.get(r["strategy_id"], 1)

        out.append(ScoredRun(
            run_id=r["run_id"],
            strategy_id=r["strategy_id"],
            strategy_name=r["strategy_name"],
            ticker=r["ticker"],
            timeframe=r["timeframe"],
            start_date=r["start_date"],
            end_date=r["end_date"],
            mode=r["mode"],
            created_at=r["created_at"],
            metrics=m,
            sharpe=sharpe,
            calmar=calmar,
            cagr=cagr,
            max_drawdown=max_dd,
            sortino=sortino,
            n_trades=n_trades,
            sharpe_score=_sharpe_score(sharpe),
            calmar_score=_calmar_score(calmar),
            consistency_score=_consistency_score(sharpe, sortino),
            dsr_score=None,  # not persisted today; frontend redistributes weight
            breadth_score=_breadth_score(distinct),
            recency_score=_recency_score(r["created_at"], now),
        ))
    return out
