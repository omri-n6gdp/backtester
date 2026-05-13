"""
Tests for the trial_sharpes_variance kwarg added to compute_deflated_sharpe
in Phase 5.7. Five targeted tests; the broader DSR tests live in
test_deflated_sharpe.py.
"""
import math

import numpy as np
import pandas as pd
import pytest

from backend.engine.robustness.deflated_sharpe import compute_deflated_sharpe


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.015, n))


# 1. Backward-compat: trial_sharpes_variance=None gives identical result.
def test_none_variance_matches_default():
    r = _make_returns()
    result_default = compute_deflated_sharpe(r, n_trials=9)
    result_none = compute_deflated_sharpe(r, n_trials=9, trial_sharpes_variance=None)
    assert result_default.dsr == result_none.dsr
    assert result_default.psr == result_none.psr


# 2. Positive empirical variance produces a different DSR than V=1/T default.
def test_empirical_variance_changes_dsr():
    r = _make_returns()
    result_default = compute_deflated_sharpe(r, n_trials=9)
    result_empirical = compute_deflated_sharpe(r, n_trials=9, trial_sharpes_variance=0.01)
    # The two should differ because 0.01 != 1/500.
    assert result_default.dsr != result_empirical.dsr


# 3. Zero variance falls back to V=1/T (treated as absent, same as default).
def test_zero_variance_falls_back_to_default():
    r = _make_returns()
    result_default = compute_deflated_sharpe(r, n_trials=9)
    result_zero = compute_deflated_sharpe(r, n_trials=9, trial_sharpes_variance=0.0)
    assert result_default.dsr == result_zero.dsr


# 4. With n_trials=1, trial_sharpes_variance is irrelevant — DSR equals PSR.
def test_variance_ignored_when_n_trials_is_1():
    r = _make_returns()
    result_no_var = compute_deflated_sharpe(r, n_trials=1)
    result_with_var = compute_deflated_sharpe(r, n_trials=1, trial_sharpes_variance=0.05)
    assert result_no_var.dsr == result_with_var.dsr
    # Both should equal PSR.
    assert math.isclose(result_no_var.dsr, result_no_var.psr, rel_tol=1e-12)


# 5. DSR is weakly monotone in trial_sharpes_variance for n_trials > 1.
#    A larger V means a higher expected-max Sharpe under H0, which is a
#    stricter benchmark, so DSR should weakly decrease as V grows.
def test_dsr_decreases_as_variance_increases():
    r = _make_returns()
    variances = [1e-6, 1e-4, 1e-2, 0.1, 1.0]
    dsrs = [
        compute_deflated_sharpe(r, n_trials=9, trial_sharpes_variance=v).dsr
        for v in variances
    ]
    for i in range(len(dsrs) - 1):
        assert dsrs[i] >= dsrs[i + 1] - 1e-12, (
            f"DSR should not increase as V grows: dsrs[{i}]={dsrs[i]:.6f}, "
            f"dsrs[{i+1}]={dsrs[i+1]:.6f}"
        )
