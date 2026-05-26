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

def test_dsr_score_null_when_not_persisted():
    """Pre-16.1 runs (or runs that didn't request DSR) keep dsr_score=None
    so the frontend redistributes that weight to available components."""
    rows = [_row()]
    s = compute_run_scores(rows, {"s1": 1}, now=_now())[0]
    assert s.dsr_score is None
    assert s.dsr is None
    assert s.sharpe_source == "full"  # no OOS persisted either


def test_dsr_score_uses_persisted_value():
    """Phase 16.1 — when a run carries a persisted DSR, the score is
    (dsr - 0.5) * 2, clipped below at 0."""
    row = _row()
    row["dsr"] = {"dsr": 0.85}
    s = compute_run_scores([row], {"s1": 1}, now=_now())[0]
    assert s.dsr == 0.85
    assert math.isclose(s.dsr_score, 0.70, rel_tol=1e-9)


def test_dsr_below_half_floors_at_zero():
    """A DSR < 0.5 means the strategy is no better than the selection-
    bias-adjusted null. Score floors at zero."""
    row = _row()
    row["dsr"] = {"dsr": 0.3}
    s = compute_run_scores([row], {"s1": 1}, now=_now())[0]
    assert s.dsr_score == 0.0


def test_sharpe_score_prefers_persisted_oos_sharpe():
    """Phase 16.1 — when OOS is stored, sharpe_score uses OOS Sharpe
    rather than full-period. The full-period value is still surfaced
    on the row for tooltips."""
    row = _row(sharpe=1.8)
    row["oos"] = {"out_of_sample": {"metrics": {"sharpe": 0.6}}}
    s = compute_run_scores([row], {"s1": 1}, now=_now())[0]
    assert s.sharpe_source == "oos"
    assert s.oos_sharpe == 0.6
    assert s.sharpe == 1.8  # raw full-period still surfaced
    # sharpe_score is OOS-based: 0.6 / 2.0 = 0.3.
    assert math.isclose(s.sharpe_score, 0.3, rel_tol=1e-9)


def test_sharpe_score_falls_back_to_full_when_no_oos():
    """Without persisted OOS, sharpe_score keeps the pre-16.1 behaviour
    (full-period Sharpe)."""
    row = _row(sharpe=1.0)
    s = compute_run_scores([row], {"s1": 1}, now=_now())[0]
    assert s.sharpe_source == "full"
    assert s.oos_sharpe is None
    assert math.isclose(s.sharpe_score, 0.5, rel_tol=1e-9)


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
