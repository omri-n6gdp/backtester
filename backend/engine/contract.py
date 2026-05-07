"""
Strategy function contract.

Every strategy in the system is a single pure function with this signature:

    def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
        ...

`data`   - OHLCV DataFrame indexed by timestamp (UTC).
           Columns: open, high, low, close, volume. Floats.
`params` - Strategy-specific parameter dict.
returns  - pd.Series of TARGET position sizes, aligned with data.index.
           Convention: -1.0 (full short) ... 0.0 (flat) ... +1.0 (full long).
           Fractional values allowed (e.g. 0.5 = half long).
           Values outside [-1, +1] are rejected by the validator.

The engine shifts positions by 1 bar internally before applying returns,
so look-ahead bias is impossible BY CONSTRUCTION even if a strategy
references the same bar's close. This is the project's single most
important guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

# Type alias for clarity at call sites.
StrategyFn = Callable[[pd.DataFrame, dict], pd.Series]


class StrategyProtocol(Protocol):
    """Structural type for a strategy function (use for type hints)."""

    def __call__(self, data: pd.DataFrame, params: dict) -> pd.Series: ...


@dataclass(frozen=True)
class StrategySpec:
    """Lightweight strategy descriptor used by the engine and registry."""
    name: str
    fn: StrategyFn
    default_params: dict


# Bounds for valid target positions. Outside this range = validator rejects.
POSITION_MIN = -1.0
POSITION_MAX = 1.0


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
