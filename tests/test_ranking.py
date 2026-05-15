"""
Unit tests for backend/engine/ranking.py — pure scoring, no I/O.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from backend.engine.ranking import compute_run_scores


def _now():
    return datetime(2026, 5, 15, tzinfo=timezone.utc)


def _row(*, run_id="r1", strategy_id="s1", ticker="SPY",
         sharpe=1.0, calmar=1.5, sortino=1.3, cagr=0.12, max_dd=-0.1,
         n_trades=10, days_old=0, mode="single", strategy_name="S1"):
    created = (_now() - timedelta(days=days_old)).isoformat()
    return {
        "run_id": run_id,
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "ticker": ticker,
        "timeframe": "1d",
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "mode": mode,
        "created_at": created,
        "metrics": {
            "sharpe": sharpe, "calmar": calmar, "sortino": sortino,
            "cagr": cagr, "max_drawdown": max_dd, "n_trades": n_trades,
        },
    }


# ---- recency decay ------------------------------------------------------

def test_recency_today_is_one():
    rows = [_row(days_old=0)]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.recency_score == 1.0


def test_recency_90_days_is_half_life():
    rows = [_row(days_old=90)]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert math.isclose(s.recency_score, math.exp(-1.0), rel_tol=1e-9)


# ---- breadth saturation -------------------------------------------------

def test_breadth_one_ticker_is_zero():
    rows = [_row()]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.breadth_score == 0.0


def test_breadth_saturates_at_6():
    rows = [_row()]
    s_low = compute_run_scores(rows, {"s1": 5}, now=_now())[0]
    s_high = compute_run_scores(rows, {"s1": 50}, now=_now())[0]
    # 50 tickers should clamp to <= 1.0, not exceed it.
    assert s_high.breadth_score == 1.0
    assert s_low.breadth_score < 1.0


# ---- core component scores ----------------------------------------------

def test_sharpe_score_saturates_at_2():
    rows = [_row(sharpe=2.5)]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.sharpe_score == 1.0


def test_negative_sharpe_floors_at_zero():
    rows = [_row(sharpe=-0.5)]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.sharpe_score == 0.0


def test_consistency_zero_when_sharpe_nonpositive():
    rows = [_row(sharpe=-0.1, sortino=1.0)]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.consistency_score == 0.0


def test_consistency_normal_case():
    # Sortino/Sharpe = 1.3 / 1.0 = 1.3; normalized = 0.65.
    rows = [_row(sharpe=1.0, sortino=1.3)]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert math.isclose(s.consistency_score, 0.65, rel_tol=1e-9)


# ---- null / missing handling -------------------------------------------

def test_dsr_score_is_always_null_for_now():
    rows = [_row()]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.dsr_score is None


def test_nan_metrics_produce_zero_scores():
    rows = [_row(sharpe=float("nan"), calmar=float("nan"))]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.sharpe_score == 0.0
    assert s.calmar_score == 0.0
    assert s.sharpe is None  # _safe_float drops NaN


# ---- ordering / stability ----------------------------------------------

def test_output_order_matches_input_order():
    """Component scoring must not reorder rows — the frontend does that."""
    rows = [
        _row(run_id="a", days_old=10, sharpe=2.0),
        _row(run_id="b", days_old=5, sharpe=0.5),
        _row(run_id="c", days_old=0, sharpe=1.0),
    ]
    scored = compute_run_scores(rows, {"s1": 1}, now=_now())
    assert [s.run_id for s in scored] == ["a", "b", "c"]
