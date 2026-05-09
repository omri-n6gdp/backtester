"""
Unit tests for backend/engine/robustness/deflated_sharpe.py.

PSR and DSR are probabilities of being above some Sharpe benchmark, so we
mostly test ordering properties (more data -> more certainty, more trials
-> lower DSR, etc.) rather than exact values, which depend on assumed
distributional moments. A few tests do hand-compute against the closed-form
PSR formula for verification.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.engine.robustness.deflated_sharpe import (
    DeflatedSharpeResult,
    _phi,
    _phi_inv,
    _psr_z_score,
    compute_deflated_sharpe,
)


def _gauss_returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    """Generate IID Gaussian returns. Standardizes the raw draws to mean 0,
    std 1 first so the realized first two moments are exactly the requested
    values. Removes seed-dependence in the moment-related tests."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    return pd.Series(raw * std + mean)


# ---------- helpers ---------------------------------------------------------

def test_phi_known_values() -> None:
    assert math.isclose(_phi(0.0), 0.5, abs_tol=1e-12)
    assert _phi(5.0) > 0.999_999
    assert _phi(-5.0) < 0.000_001


def test_phi_inv_round_trips() -> None:
    for p in [0.001, 0.05, 0.5, 0.9, 0.999]:
        assert math.isclose(_phi(_phi_inv(p)), p, abs_tol=1e-9)


def test_phi_inv_clamps_at_boundary() -> None:
    lo = _phi_inv(0.0)
    hi = _phi_inv(1.0)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < -7.0 and hi > 7.0


# ---------- input validation ------------------------------------------------

def test_n_trials_must_be_at_least_one() -> None:
    r = _gauss_returns(100, mean=0.0, std=0.01)
    with pytest.raises(ValueError, match="n_trials"):
        compute_deflated_sharpe(r, n_trials=0)
    with pytest.raises(ValueError, match="n_trials"):
        compute_deflated_sharpe(r, n_trials=-3)


def test_too_short_returns_returns_all_none() -> None:
    r = pd.Series([0.01])
    out = compute_deflated_sharpe(r, n_trials=5)
    assert out.psr is None and out.dsr is None
    assert out.sharpe_annualized is None
    assert out.skew is None and out.excess_kurtosis is None


def test_zero_variance_returns_none_psr() -> None:
    r = pd.Series([0.005] * 100)
    out = compute_deflated_sharpe(r, n_trials=1)
    assert out.psr is None and out.dsr is None
    assert out.sharpe_per_bar == 0.0


def test_nan_in_returns_is_dropped() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 100))
    r.iloc[5] = float("nan")
    r.iloc[10] = float("nan")
    out = compute_deflated_sharpe(r, n_trials=1)
    assert out.n_returns == 98


# ---------- PSR semantics ---------------------------------------------------

def test_psr_zero_sharpe_is_half() -> None:
    r = _gauss_returns(500, mean=0.0, std=0.01, seed=0)
    out = compute_deflated_sharpe(r, n_trials=1)
    assert abs(out.psr - 0.5) < 0.05


def test_psr_high_sharpe_normal_is_near_one() -> None:
    r = _gauss_returns(1000, mean=0.001, std=0.01, seed=1)
    out = compute_deflated_sharpe(r, n_trials=1)
    assert out.psr > 0.99


def test_psr_negative_sharpe_is_near_zero() -> None:
    r = _gauss_returns(1000, mean=-0.001, std=0.01, seed=2)
    out = compute_deflated_sharpe(r, n_trials=1)
    assert out.psr < 0.01


def test_psr_more_data_higher_certainty_for_same_sharpe() -> None:
    """Larger T -> tighter PSR around the truth -> higher PSR for true SR>0."""
    short = _gauss_returns(100, mean=0.0005, std=0.01, seed=3)
    longer = _gauss_returns(1000, mean=0.0005, std=0.01, seed=3)
    psr_short = compute_deflated_sharpe(short, n_trials=1).psr
    psr_long = compute_deflated_sharpe(longer, n_trials=1).psr
    assert psr_long > psr_short


def test_psr_negative_skew_lowers_psr_at_same_sharpe() -> None:
    """Closed-form: matched per-bar Sharpe / kurtosis but flipped skew sign.
    Negative skew -> larger Sharpe-estimator variance -> smaller z -> lower PSR.
    """
    sharpe_per_bar = 0.05
    full_kurt = 3.0
    n = 500
    z_neg = _psr_z_score(sharpe_per_bar, skew=-2.0, full_kurtosis=full_kurt, n_returns=n, sr_benchmark=0.0)
    z_zero = _psr_z_score(sharpe_per_bar, skew=0.0, full_kurtosis=full_kurt, n_returns=n, sr_benchmark=0.0)
    assert _phi(z_neg) < _phi(z_zero)


def test_psr_fat_tails_lower_psr_at_same_sharpe() -> None:
    """Higher excess kurtosis raises Sharpe-estimator variance -> lower PSR."""
    sharpe_per_bar = 0.05
    n = 500
    z_normal = _psr_z_score(sharpe_per_bar, skew=0.0, full_kurtosis=3.0, n_returns=n, sr_benchmark=0.0)
    z_fat = _psr_z_score(sharpe_per_bar, skew=0.0, full_kurtosis=10.0, n_returns=n, sr_benchmark=0.0)
    assert _phi(z_fat) < _phi(z_normal)


def test_psr_sample_negative_skew_lowers_psr() -> None:
    """End-to-end version: a negatively-skewed sample should yield lower PSR
    than a Gaussian sample with identical per-bar Sharpe."""
    rng = np.random.default_rng(0)
    n = 1000
    target_mean, target_std = 0.0005, 0.01
    gaussian = _gauss_returns(n, target_mean, target_std, seed=10)

    raw = rng.normal(0.0, 1.0, n)
    raw[:30] -= 3.0  # cluster of large negatives -> negative skew
    skewed = pd.Series((raw - raw.mean()) / raw.std(ddof=1) * target_std + target_mean)

    psr_g = compute_deflated_sharpe(gaussian, n_trials=1).psr
    psr_s = compute_deflated_sharpe(skewed, n_trials=1).psr
    sg = compute_deflated_sharpe(gaussian, n_trials=1).sharpe_per_bar
    ss = compute_deflated_sharpe(skewed, n_trials=1).sharpe_per_bar
    assert math.isclose(sg, ss, rel_tol=1e-8)
    assert pd.Series(skewed).skew() < 0
    assert psr_s < psr_g


# ---------- DSR semantics ---------------------------------------------------

def test_dsr_n_trials_1_equals_psr() -> None:
    r = _gauss_returns(500, mean=0.0005, std=0.01, seed=4)
    out = compute_deflated_sharpe(r, n_trials=1)
    assert out.psr is not None and out.dsr is not None
    assert math.isclose(out.psr, out.dsr, rel_tol=1e-12)
    assert out.expected_max_sharpe_annualized == 0.0


def test_dsr_decreases_with_more_trials() -> None:
    r = _gauss_returns(1000, mean=0.0005, std=0.01, seed=5)
    dsr_values = []
    for n_trials in [1, 5, 25, 100, 1000]:
        out = compute_deflated_sharpe(r, n_trials=n_trials)
        dsr_values.append(out.dsr)
    for a, b in zip(dsr_values, dsr_values[1:]):
        assert b < a


def test_expected_max_sharpe_grows_with_n_trials() -> None:
    r = _gauss_returns(1000, mean=0.0, std=0.01, seed=6)
    e_values = []
    for n_trials in [2, 10, 100, 1000]:
        out = compute_deflated_sharpe(r, n_trials=n_trials)
        e_values.append(out.expected_max_sharpe_annualized)
    for a, b in zip(e_values, e_values[1:]):
        assert b > a


def test_dsr_in_unit_interval() -> None:
    r = _gauss_returns(500, mean=0.0005, std=0.01, seed=7)
    for n_trials in [1, 2, 50, 5000]:
        out = compute_deflated_sharpe(r, n_trials=n_trials)
        assert 0.0 <= out.psr <= 1.0
        assert 0.0 <= out.dsr <= 1.0


def test_dsr_high_sharpe_remains_high_under_modest_deflation() -> None:
    """A genuinely strong strategy (high SR, lots of data) should still
    produce a high DSR even after deflating for hundreds of trials."""
    r = _gauss_returns(2000, mean=0.0015, std=0.01, seed=8)
    out = compute_deflated_sharpe(r, n_trials=100)
    assert out.dsr > 0.95


def test_dsr_borderline_sharpe_collapses_under_heavy_deflation() -> None:
    """A weak strategy that looks good only against a zero benchmark should
    have its DSR fall sharply when many trials are accounted for."""
    r = _gauss_returns(500, mean=0.0003, std=0.01, seed=9)
    psr = compute_deflated_sharpe(r, n_trials=1).psr
    dsr_many = compute_deflated_sharpe(r, n_trials=10_000).dsr
    assert dsr_many < psr - 0.2


# ---------- closed-form sanity ----------------------------------------------

def test_psr_matches_explicit_formula_for_normal() -> None:
    """PSR for a near-Gaussian sample should match the explicit closed form
    Phi(SR * sqrt((T-1) / variance_term)) using the actual sample skew/kurt."""
    n = 500
    r = _gauss_returns(n, mean=0.0005, std=0.01, seed=11)
    out = compute_deflated_sharpe(r, n_trials=1)
    skew = out.skew
    excess_kurt = out.excess_kurtosis
    sr = out.sharpe_per_bar
    var_term = 1.0 - skew * sr + ((excess_kurt + 3.0) - 1.0) / 4.0 * sr ** 2
    z_expected = sr * math.sqrt(n - 1) / math.sqrt(var_term)
    psr_expected = _phi(z_expected)
    assert math.isclose(out.psr, psr_expected, rel_tol=1e-10)


def test_full_kurtosis_3_for_normal() -> None:
    """Sanity check: pandas .kurt() returns excess kurt; engine adds 3
    internally to get full kurtosis. So a Gaussian sample should have
    excess kurt ~= 0."""
    r = _gauss_returns(2000, mean=0.0, std=0.01, seed=12)
    out = compute_deflated_sharpe(r, n_trials=1)
    assert abs(out.excess_kurtosis) < 0.3


def test_known_sharpe_per_bar_matches_metrics_module_convention() -> None:
    """sharpe_per_bar = mean / std (no annualization). sharpe_annualized
    must equal that times sqrt(bars_per_year). This matches metrics.sharpe()
    when risk_free=0."""
    r = _gauss_returns(1000, mean=0.001, std=0.01, seed=13)
    out = compute_deflated_sharpe(r, n_trials=1, bars_per_year=252)
    expected_pb = float(r.mean()) / float(r.std(ddof=1))
    assert math.isclose(out.sharpe_per_bar, expected_pb, rel_tol=1e-12)
    assert math.isclose(out.sharpe_annualized, expected_pb * math.sqrt(252), rel_tol=1e-12)
