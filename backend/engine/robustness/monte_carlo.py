"""
Monte Carlo trade-order shuffling.

Given a BacktestResult, randomly reorder the trade-level returns N times
and reconstruct an equity curve for each shuffled ordering. Reports
distributional statistics over the shuffles to answer:

    "Was this strategy's equity curve a function of skill (the bag of
    trade returns) or of luck (the *order* in which those returns
    happened to occur)?"

Mathematical notes
------------------
- Final equity is INVARIANT under trade-order shuffling, because
  prod(1 + r_i) is permutation-invariant. The `final_equities` array
  is therefore a delta spike at the original final equity (within
  floating-point noise). It's still computed and exposed as a sanity
  check; tests assert the invariance holds to high precision.
- CAGR is likewise invariant (it's a function of total return and
  total time span, both invariant under shuffling). We deliberately
  do not expose a CAGR distribution because there is none.
- Max drawdown is HIGHLY ORDER-DEPENDENT. A strategy whose worst trades
  cluster at the start has a far deeper DD than one whose worst trades
  are interspersed. The MC max-DD distribution is the meaningful output.

Outputs
-------
- Per-iteration scalars: final equity, max drawdown, max DD duration
  (in trades, not bars — there are no flat-bars between trades on the
  shuffled curve).
- Equity bands: P{q} of equity at each trade-index k (k = 0..n_trades).
- Drawdown bands: same shape, for the running drawdown.
- Original-DD percentile: where the unshuffled (actual) max DD ranks
  in the shuffled distribution. High = original was a lucky ordering;
  low = original was an unusually bad ordering.

This module deliberately type-hints `BacktestResult` only under
`TYPE_CHECKING`, so the runtime accepts any object with the duck-typed
shape (`.equity` -> Series-like, `.trades` -> iterable of objects with
`.pnl_return` floats). This keeps the unit tests free of engine
dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.engine.vectorized import BacktestResult  # pragma: no cover


@dataclass(frozen=True)
class MonteCarloConfig:
    n_iterations: int = 1000
    seed: int | None = 42
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be >= 1, got {self.n_iterations}"
            )
        if not self.quantiles:
            raise ValueError("quantiles must be a non-empty sequence")
        for q in self.quantiles:
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"quantile must be in [0, 1], got {q}")


@dataclass
class MonteCarloResult:
    """Distributional output of `monte_carlo_shuffle`."""

    n_iterations: int
    n_trades: int
    seed: int | None
    quantiles: list[float]

    # Reference values from the original (unshuffled) ordering.
    original_final_equity: float
    original_max_drawdown: float
    original_max_drawdown_trades: int

    # Per-iteration scalars (length = n_iterations).
    final_equities: list[float]
    max_drawdowns: list[float]
    max_drawdown_durations: list[int]

    # Equity / drawdown bands at each trade index.
    # Each key is "q{int(q*100):02d}", e.g. "q05", "q50", "q95".
    # Each list has length n_trades + 1 (includes the starting bar).
    equity_bands: dict[str, list[float]]
    drawdown_bands: dict[str, list[float]]

    # Distribution of max drawdown across shuffles.
    max_dd_quantiles: dict[str, float]

    # Where the original max DD ranks in the shuffled distribution.
    # Convention: percentile = 100 * fraction of shuffles whose max DD
    # was DEEPER (more negative) than the original.
    #   100 = original was the luckiest ordering (no shuffle was worse)
    #     0 = original was the unluckiest (no shuffle was deeper)
    # Returned as float("nan") when n_trades == 0.
    original_dd_percentile: float


def _quantile_key(q: float) -> str:
    """Stable string key for a quantile, e.g. 0.05 -> 'q05'."""
    return f"q{int(round(q * 100)):02d}"


def _drawdown_duration_in_trades(dd_row: np.ndarray) -> int:
    """Longest contiguous run of bars where dd_row < 0."""
    if dd_row.size == 0:
        return 0
    underwater = dd_row < 0.0
    if not underwater.any():
        return 0
    best = cur = 0
    for v in underwater:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def monte_carlo_shuffle(
    result: "BacktestResult",
    config: MonteCarloConfig | None = None,
) -> MonteCarloResult:
    """Shuffle the trade-return sequence N times and report distributions."""
    cfg = config or MonteCarloConfig()
    quantile_list: list[float] = list(cfg.quantiles)

    starting_capital = float(result.equity.iloc[0]) if len(result.equity) else 0.0

    trade_returns = np.array(
        [float(t.pnl_return) for t in result.trades], dtype=float
    )
    n_trades = int(trade_returns.size)
    n_iter = int(cfg.n_iterations)

    if n_trades == 0:
        # Degenerate: no trades to shuffle. Return zeroed bands at the
        # starting capital so downstream code (UI) never crashes.
        return MonteCarloResult(
            n_iterations=n_iter,
            n_trades=0,
            seed=cfg.seed,
            quantiles=quantile_list,
            original_final_equity=starting_capital,
            original_max_drawdown=0.0,
            original_max_drawdown_trades=0,
            final_equities=[starting_capital] * n_iter,
            max_drawdowns=[0.0] * n_iter,
            max_drawdown_durations=[0] * n_iter,
            equity_bands={_quantile_key(q): [starting_capital] for q in quantile_list},
            drawdown_bands={_quantile_key(q): [0.0] for q in quantile_list},
            max_dd_quantiles={_quantile_key(q): 0.0 for q in quantile_list},
            original_dd_percentile=float("nan"),
        )

    # --- Original (unshuffled) reference curve ---------------------------
    orig_equity_inner = starting_capital * np.cumprod(1.0 + trade_returns)
    orig_equity = np.concatenate(([starting_capital], orig_equity_inner))
    orig_running_max = np.maximum.accumulate(orig_equity)
    orig_dd = (orig_equity - orig_running_max) / orig_running_max
    orig_max_dd = float(orig_dd.min())
    orig_max_dd_dur = _drawdown_duration_in_trades(orig_dd)
    orig_final = float(orig_equity[-1])

    # --- Shuffles --------------------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    # Build a (n_iter, n_trades) matrix of permutations. We loop because
    # rng.permuted with axis= is finicky across NumPy versions; this is
    # cheap (n_iter * n_trades is well under a million in normal use).
    shuffled = np.empty((n_iter, n_trades), dtype=float)
    for i in range(n_iter):
        shuffled[i] = rng.permutation(trade_returns)

    gross = 1.0 + shuffled
    eq_inner = starting_capital * np.cumprod(gross, axis=1)
    starts = np.full((n_iter, 1), starting_capital, dtype=float)
    eq_full = np.concatenate([starts, eq_inner], axis=1)  # (n_iter, n_trades+1)

    running_max = np.maximum.accumulate(eq_full, axis=1)
    dd_full = (eq_full - running_max) / running_max  # always <= 0

    final_equities = eq_full[:, -1]
    max_drawdowns = dd_full.min(axis=1)
    max_dd_durations = np.fromiter(
        (_drawdown_duration_in_trades(dd_full[i]) for i in range(n_iter)),
        dtype=int,
        count=n_iter,
    )

    # Bands at each step
    eq_quantiles = np.quantile(eq_full, q=quantile_list, axis=0)
    dd_quantiles = np.quantile(dd_full, q=quantile_list, axis=0)
    equity_bands = {
        _quantile_key(q): eq_quantiles[i].tolist()
        for i, q in enumerate(quantile_list)
    }
    drawdown_bands = {
        _quantile_key(q): dd_quantiles[i].tolist()
        for i, q in enumerate(quantile_list)
    }

    # Quantiles of the max-DD distribution (a single number per q).
    max_dd_qvals = np.quantile(max_drawdowns, q=quantile_list)
    max_dd_quantiles = {
        _quantile_key(q): float(max_dd_qvals[i])
        for i, q in enumerate(quantile_list)
    }

    # Original DD percentile: fraction of shuffles whose DD was DEEPER
    # (more negative) than the original. High = original was lucky.
    orig_dd_pct = float(100.0 * np.mean(max_drawdowns < orig_max_dd))

    return MonteCarloResult(
        n_iterations=n_iter,
        n_trades=n_trades,
        seed=cfg.seed,
        quantiles=quantile_list,
        original_final_equity=orig_final,
        original_max_drawdown=orig_max_dd,
        original_max_drawdown_trades=orig_max_dd_dur,
        final_equities=final_equities.tolist(),
        max_drawdowns=max_drawdowns.tolist(),
        max_drawdown_durations=max_dd_durations.tolist(),
        equity_bands=equity_bands,
        drawdown_bands=drawdown_bands,
        max_dd_quantiles=max_dd_quantiles,
        original_dd_percentile=orig_dd_pct,
    )
