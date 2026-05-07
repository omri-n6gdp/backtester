"""
Smoke test: SMA(50)/SMA(200) crossover on SPY, 2010 to today.

Prints all metrics. Saves an equity-curve PNG.

Run:
    python scripts/smoke_backtest.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Ensure the project root is importable when running as `python scripts/foo.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from backend.data.prices import get_prices
from backend.engine.costs import DEFAULT_COST_CONFIG
from backend.engine.metrics import compute_metrics
from backend.engine.vectorized import run_backtest


def sma_crossover_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    """Long when SMA(short) > SMA(long), flat otherwise. Long-only."""
    short = int(params.get("short", 50))
    long = int(params.get("long", 200))
    sma_s = data["close"].rolling(short, min_periods=short).mean()
    sma_l = data["close"].rolling(long, min_periods=long).mean()
    pos = (sma_s > sma_l).astype(float)  # 0 or 1
    return pos


def _print_metrics(name: str, metrics_dict: dict) -> None:
    print(f"\n=== {name} ===")
    groups = {
        "Return":      ["total_return", "cagr", "best_month", "worst_month"],
        "Risk":        ["annualized_vol", "max_drawdown", "max_drawdown_duration_days",
                        "avg_drawdown", "ulcer_index", "downside_deviation",
                        "var_95", "var_99", "cvar_95"],
        "Risk-adj":    ["sharpe", "sortino", "calmar", "omega", "mar"],
        "Trades":      ["n_trades", "win_rate", "profit_factor", "expectancy",
                        "avg_win", "avg_loss", "avg_holding_bars",
                        "max_consecutive_wins", "max_consecutive_losses"],
    }
    for group, keys in groups.items():
        print(f"  [{group}]")
        for k in keys:
            v = metrics_dict.get(k)
            if isinstance(v, float):
                print(f"    {k:32s} {v:>12.4f}")
            else:
                print(f"    {k:32s} {v:>12}")


def main() -> None:
    print("Fetching SPY daily 2010-01-01 to today...")
    data = get_prices("SPY", "1d", date(2010, 1, 1), date.today())
    print(f"  {len(data)} bars from {data.index.min().date()} to {data.index.max().date()}")

    print("\nRunning SMA(50)/SMA(200) crossover backtest...")
    result = run_backtest(
        sma_crossover_strategy,
        data,
        params={"short": 50, "long": 200},
        cost_config=DEFAULT_COST_CONFIG,
    )
    metrics = compute_metrics(result)
    _print_metrics("SMA(50)/SMA(200) on SPY", metrics.to_dict())

    print(f"\nFinal equity: ${result.equity.iloc[-1]:,.2f}")
    print(f"Starting:     ${result.starting_capital:,.2f}")
    print(f"Trades:       {len(result.trades)}")

    # Save an equity-curve PNG (matplotlib used here as a dev convenience).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = Path(__file__).resolve().parent.parent / "smoke_equity.png"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(result.equity.index, result.equity.values, label="Strategy", linewidth=1.2)
        # Buy-and-hold reference.
        bh = (1 + data["close"].pct_change().fillna(0)).cumprod() * result.starting_capital
        ax1.plot(bh.index, bh.values, label="Buy & Hold", linewidth=1.0, alpha=0.6)
        ax1.set_title("SMA(50)/SMA(200) on SPY — equity curve")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.fill_between(result.drawdown.index, result.drawdown.values, 0, alpha=0.4)
        ax2.set_title("Drawdown")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        print(f"\nSaved equity curve PNG: {out_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping PNG. `pip install matplotlib` to enable.)")


if __name__ == "__main__":
    main()
