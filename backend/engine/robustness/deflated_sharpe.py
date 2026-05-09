"""
Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

Two corrections applied on top of the raw Sharpe:

1. PSR (Bailey & López de Prado, 2012) accounts for non-normal returns:
   negative skew and fat tails make the Sharpe estimator noisier than the
   IID-Gaussian formula assumes, so the *certainty* that the true Sharpe
   is positive is lower than the raw number suggests.

2. DSR (Bailey & López de Prado, 2014) adds a correction for selection
   bias: when the reported Sharpe is the best of N trials (parameter
   combos, asset/timeframe combos, anything optimized over), the best-of-N
   has an upward expected value EVEN UNDER A ZERO-TRUE-SHARPE NULL.
   DSR asks "is the observed Sharpe statistically distinguishable from
   the best-of-N you'd get from sheer luck?"

Both PSR and DSR are PROBABILITIES in [0, 1]:
   PSR  = P(true SR > 0          | SR_hat, skew, kurt, T)
   DSR  = P(true SR > E[max|H0]  | SR_hat, skew, kurt, T, N)

Math
----
Asymptotic variance of the per-bar Sharpe estimator (Lo, 2002):

    Var(SR_hat) ~= [1 - g3*SR + ((g4 - 1)/4)*SR^2] / (T - 1)

where g3 = sample skew, g4 = sample FULL kurtosis (= 3 for Gaussian),
T = number of returns. Note pandas.Series.kurt() returns EXCESS kurtosis
(Fisher's, = 0 for Gaussian), so we add 3.

PSR(SR*) = Phi((SR_hat - SR*) * sqrt(T - 1) /
                sqrt(1 - g3*SR_hat + (g4 - 1)/4 * SR_hat^2))

For DSR, SR* is set to the expected best-of-N Sharpe under H0:true_SR=0.
Bailey & Lopez de Prado (2014), eq. 9, using Gumbel-distribution
order-statistic asymptotics:

    SR*_0 ~= sqrt(V) * ((1 - ge)*Phi^-1(1 - 1/N) + ge*Phi^-1(1 - 1/(N*e)))

where ge ~= 0.5772 is the Euler-Mascheroni constant, e is Napier's number,
and V is the variance of per-trial Sharpes. We don't have all N trial
Sharpes (we ran one backtest), so we use the conservative asymptotic null
estimate V ~= 1/T (each Sharpe under H0 is ~Normal(0, 1/T) for IID
Gaussian returns). Phase 5.7's sweep route will replace this with an
empirical V from the actual trial Sharpes.

Units note
----------
All internal math is per-bar. We expose annualized SR and SR*_0 (= per-bar
* sqrt(bars_per_year)) for human-readable display alongside.

This module deliberately avoids scipy (matches project policy of minimal
dependencies). Phi and Phi^-1 come from stdlib's `statistics.NormalDist`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from backend.engine.vectorized import BacktestResult  # pragma: no cover


_NORM = NormalDist()
_GAMMA_EULER_MASCHERONI = 0.5772156649015329


def _phi(x: float) -> float:
    """Standard normal CDF Phi."""
    return _NORM.cdf(x)


def _phi_inv(p: float) -> float:
    """Standard normal inverse CDF Phi^-1. Clamps at boundary to avoid +/-inf."""
    eps = 1e-15
    p_clamped = min(max(p, eps), 1.0 - eps)
    return _NORM.inv_cdf(p_clamped)


@dataclass
class DeflatedSharpeResult:
    """Output of `compute_deflated_sharpe()`. NaNs reported as None."""

    n_returns: int
    n_trials: int
    bars_per_year: int

    # Observed values (annualized for display).
    sharpe_annualized: float | None
    sharpe_per_bar: float | None
    skew: float | None
    excess_kurtosis: float | None

    # Expected best-of-N Sharpe under H0:true_SR=0 (annualized).
    expected_max_sharpe_annualized: float | None

    # Probabilities (in [0, 1]).
    psr: float | None             # P(true SR > 0)
    dsr: float | None             # P(true SR > E[max|H0])


def _psr_z_score(
    sharpe_per_bar: float,
    skew: float,
    full_kurtosis: float,
    n_returns: int,
    sr_benchmark: float,
) -> float:
    """Z-score input to Phi for PSR computation. Pure helper."""
    var_term = (
        1.0
        - skew * sharpe_per_bar
        + ((full_kurtosis - 1.0) / 4.0) * sharpe_per_bar ** 2
    )
    # Pathological distributions (very low kurtosis at extreme Sharpe) can
    # make this variance-of-Sharpe estimator non-positive. Clamp to a tiny
    # positive number so downstream sqrt is finite.
    var_term = max(var_term, 1e-12)
    return (sharpe_per_bar - sr_benchmark) * math.sqrt(n_returns - 1) / math.sqrt(var_term)


def compute_deflated_sharpe(
    returns: pd.Series,
    n_trials: int = 1,
    bars_per_year: int = 252,
) -> DeflatedSharpeResult:
    """Compute PSR and DSR for a return series.

    Args
    ----
    returns:
        per-bar strategy returns (typically `BacktestResult.returns`).
    n_trials:
        number of independent trials (parameter combos, asset/timeframe
        combos, anything optimized over) considered when arriving at this
        strategy. n_trials=1 means "no selection bias", so DSR collapses
        to PSR. n_trials < 1 raises ValueError.
    bars_per_year:
        used only for annualized-SR display; the math is in per-bar units.

    Returns
    -------
    DeflatedSharpeResult. When the input is too short or has zero variance,
    fields are None rather than NaN.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    # Sanitize: drop NaN, ensure numeric.
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = int(len(r))
    if n < 2:
        return DeflatedSharpeResult(
            n_returns=n, n_trials=n_trials, bars_per_year=bars_per_year,
            sharpe_annualized=None, sharpe_per_bar=None,
            skew=None, excess_kurtosis=None,
            expected_max_sharpe_annualized=None,
            psr=None, dsr=None,
        )

    # A genuinely constant series has std() that is mathematically 0 but
    # may be a tiny floating-point artifact (~1e-18) due to two-pass numerical
    # algorithms. Detect "effectively constant" semantically rather than
    # relying on `sd == 0.0`.
    if r.nunique() <= 1:
        return DeflatedSharpeResult(
            n_returns=n, n_trials=n_trials, bars_per_year=bars_per_year,
            sharpe_annualized=0.0, sharpe_per_bar=0.0,
            skew=None, excess_kurtosis=None,
            expected_max_sharpe_annualized=None,
            psr=None, dsr=None,
        )

    sd = float(r.std(ddof=1))
    if math.isnan(sd) or sd == 0.0:
        # Defensive: any other path to zero/NaN std (shouldn't reach here
        # given the nunique check above, but kept for safety).
        return DeflatedSharpeResult(
            n_returns=n, n_trials=n_trials, bars_per_year=bars_per_year,
            sharpe_annualized=0.0, sharpe_per_bar=0.0,
            skew=None, excess_kurtosis=None,
            expected_max_sharpe_annualized=None,
            psr=None, dsr=None,
        )

    mean = float(r.mean())
    sharpe_per_bar = mean / sd
    sharpe_ann = sharpe_per_bar * math.sqrt(bars_per_year)

    # pandas: Fisher's (excess) kurt; needs n>=4. .skew() needs n>=3.
    # When NaN-out occurs we fall back to Gaussian assumption (skew=0,
    # excess_kurt=0) for the math but expose None to signal the fallback.
    skew_raw = float(r.skew()) if n >= 3 else float("nan")
    kurt_raw = float(r.kurt()) if n >= 4 else float("nan")
    skew_for_math = 0.0 if math.isnan(skew_raw) else skew_raw
    excess_kurt_for_math = 0.0 if math.isnan(kurt_raw) else kurt_raw
    full_kurt = excess_kurt_for_math + 3.0

    psr_z = _psr_z_score(sharpe_per_bar, skew_for_math, full_kurt, n, sr_benchmark=0.0)
    psr = _phi(psr_z)

    if n_trials == 1:
        # No selection bias to deflate against. DSR = PSR by definition.
        sr_star_per_bar = 0.0
        dsr = psr
    else:
        # Conservative null variance V ~= 1/T (each Sharpe ~ Normal(0, 1/T)
        # under H0:true_SR=0 with IID Gaussian returns).
        v = 1.0 / n
        order_stat_term = (
            (1.0 - _GAMMA_EULER_MASCHERONI) * _phi_inv(1.0 - 1.0 / n_trials)
            + _GAMMA_EULER_MASCHERONI * _phi_inv(1.0 - 1.0 / (n_trials * math.e))
        )
        sr_star_per_bar = math.sqrt(v) * order_stat_term
        dsr_z = _psr_z_score(
            sharpe_per_bar, skew_for_math, full_kurt, n,
            sr_benchmark=sr_star_per_bar,
        )
        dsr = _phi(dsr_z)

    sr_star_ann = sr_star_per_bar * math.sqrt(bars_per_year)

    return DeflatedSharpeResult(
        n_returns=n,
        n_trials=n_trials,
        bars_per_year=bars_per_year,
        sharpe_annualized=sharpe_ann,
        sharpe_per_bar=sharpe_per_bar,
        skew=None if math.isnan(skew_raw) else skew_raw,
        excess_kurtosis=None if math.isnan(kurt_raw) else kurt_raw,
        expected_max_sharpe_annualized=sr_star_ann,
        psr=psr,
        dsr=dsr,
    )
