"""Robustness analysis layer (Phase 5).

Modules in this package are added across Phase 5:
    oos             — in-sample / out-of-sample split (5.1)
    walk_forward    — rolling-window optimization (5.2)
    monte_carlo     — trade-shuffling confidence bands (5.3)
    deflated_sharpe — multiple-comparisons-corrected Sharpe (5.4)
    regimes         — regime-conditional metrics (5.5)
    heatmaps        — 2D parameter sweep visualization (5.6)

Public re-exports follow.
"""

from .oos import OOSResult, WindowResult, split_result
from .walk_forward import (
    WalkForwardResult,
    WindowReport,
    walk_forward,
)

__all__ = [
    "OOSResult",
    "WindowResult",
    "split_result",
    "WalkForwardResult",
    "WindowReport",
    "walk_forward",
]
