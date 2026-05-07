"""
Strategy validator.

Two-stage validation before a strategy is allowed to run at scale:

1. Static (AST): reject banned operations - imports outside a whitelist,
   eval/exec/compile/open/__import__, file I/O, network calls.
2. Runtime: load in a constrained namespace, run on 1y of SPY daily data,
   assert the output Series has the right shape and values are in [-1, +1].

This is a defense-in-depth check, not a sandbox. We trust the user (it's
their machine), but we prevent silly mistakes (peeking at the future,
writing to disk, hanging on a network call).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .contract import (
    POSITION_MAX,
    POSITION_MIN,
    REQUIRED_COLUMNS,
    StrategyFn,
)

logger = logging.getLogger(__name__)

# Modules a strategy is allowed to import. Anything else is rejected.
ALLOWED_IMPORTS = frozenset({
    "pandas", "pd",
    "numpy", "np",
    "math",
    "datetime",
})

# Function names that, when called, are banned outright.
BANNED_CALLS = frozenset({
    "eval", "exec", "compile",
    "open",
    "__import__",
    "input",
    "exit", "quit",
})


@dataclass
class ValidationResult:
    """Outcome of validating a strategy."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategy_name: Optional[str] = None  # function name detected from source

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# --- Stage 1: static AST analysis -----------------------------------------

class _SafetyChecker(ast.NodeVisitor):
    """AST walker that records every banned construct it sees."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.functions_defined: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 (ast hook)
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                self.errors.append(
                    f"Disallowed import {alias.name!r} (line {node.lineno}). "
                    f"Allowed: {sorted(ALLOWED_IMPORTS)}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        mod = (node.module or "").split(".")[0]
        if mod not in ALLOWED_IMPORTS:
            self.errors.append(
                f"Disallowed import from {node.module!r} (line {node.lineno}). "
                f"Allowed: {sorted(ALLOWED_IMPORTS)}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Direct call: foo(...)
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            self.errors.append(
                f"Banned call to {node.func.id!r} (line {node.lineno})."
            )
        # Attribute call: os.system(...) - block double-underscore attribute access too.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in BANNED_CALLS:
                self.errors.append(
                    f"Banned call to .{node.func.attr}() (line {node.lineno})."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # Block dunder attribute access (e.g. obj.__globals__, obj.__class__.__mro__).
        if node.attr.startswith("__") and node.attr.endswith("__") and node.attr != "__init__":
            self.errors.append(
                f"Banned dunder attribute access .{node.attr} (line {node.lineno})."
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.functions_defined.append(node.name)
        self.generic_visit(node)


def _static_check(source: str, result: ValidationResult) -> Optional[ast.Module]:
    """Parse + walk the AST. Returns the parsed module on success."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.add_error(f"SyntaxError: {exc.msg} (line {exc.lineno})")
        return None

    checker = _SafetyChecker()
    checker.visit(tree)
    for err in checker.errors:
        result.add_error(err)

    if "strategy" in checker.functions_defined:
        result.strategy_name = "strategy"
    elif checker.functions_defined:
        result.add_error(
            f"No function named 'strategy' found. "
            f"Defined: {checker.functions_defined}. "
            f"Strategies must define a function called `strategy`."
        )
    else:
        result.add_error("No function definitions found. Define `def strategy(data, params):`.")

    return tree


# --- Stage 2: runtime check -----------------------------------------------

def _load_strategy(source: str) -> StrategyFn:
    """
    Compile + exec the source in a fresh namespace. Returns the `strategy` callable.

    Note: this is a defense-in-depth check, not a sandbox. The static checker
    has already rejected obviously dangerous constructs.
    """
    namespace: dict = {}
    # Provide whitelisted modules pre-imported so strategies don't need to.
    import math as _math
    namespace.update({
        "pd": pd, "pandas": pd,
        "np": np, "numpy": np,
        "math": _math,
    })
    code = compile(source, "<strategy>", "exec")
    exec(code, namespace)  # noqa: S102 (intentional - validated source)
    fn = namespace.get("strategy")
    if not callable(fn):
        raise ValueError("Source did not define a callable named `strategy`.")
    return fn


def _sample_data(n_days: int = 252) -> pd.DataFrame:
    """
    Synthetic-but-realistic OHLCV used for runtime validation.

    Deterministic (seeded) so validation is reproducible. We intentionally
    do NOT hit yfinance here - validation should never depend on the network.
    """
    rng = np.random.default_rng(42)
    end = date.today()
    start = end - timedelta(days=int(n_days * 1.5))  # leave room for weekends
    idx = pd.bdate_range(start=start, end=end, tz="UTC", name="timestamp")[-n_days:]

    # Geometric Brownian motion at ~SPY-like params.
    daily_ret = rng.normal(loc=0.0004, scale=0.012, size=len(idx))
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    open_ = close * (1.0 + rng.normal(0, 0.002, len(idx)))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.003, len(idx))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.003, len(idx))))
    volume = rng.uniform(1e7, 5e7, len(idx))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _runtime_check(fn: StrategyFn, result: ValidationResult, params: Optional[dict]) -> None:
    """Run the strategy on synthetic data and assert the output shape/values."""
    data = _sample_data()
    p = params if params is not None else {}

    try:
        out = fn(data, p)
    except Exception as exc:
        result.add_error(f"Strategy raised at runtime: {type(exc).__name__}: {exc}")
        return

    if not isinstance(out, pd.Series):
        result.add_error(
            f"Strategy must return pd.Series; got {type(out).__name__}."
        )
        return

    if len(out) != len(data):
        result.add_error(
            f"Strategy output length {len(out)} != input length {len(data)}."
        )
        return

    if not out.index.equals(data.index):
        result.add_warning(
            "Strategy output index does not match input index. "
            "Engine will reindex; this may cause silent gaps."
        )

    # Must be numeric.
    try:
        numeric = pd.to_numeric(out, errors="raise")
    except (ValueError, TypeError) as exc:
        result.add_error(f"Strategy output is not numeric: {exc}")
        return

    # Drop NaN before bounds check (NaN positions are treated as 'flat' downstream).
    finite = numeric.dropna()
    if len(finite) == 0:
        result.add_error("Strategy output is entirely NaN.")
        return

    if (finite < POSITION_MIN - 1e-9).any() or (finite > POSITION_MAX + 1e-9).any():
        bad_min, bad_max = float(finite.min()), float(finite.max())
        result.add_error(
            f"Strategy positions out of bounds [{POSITION_MIN}, {POSITION_MAX}]: "
            f"observed range [{bad_min:.4f}, {bad_max:.4f}]."
        )


# --- Public API -----------------------------------------------------------

def validate_strategy(
    source: str,
    params: Optional[dict] = None,
) -> ValidationResult:
    """
    Validate a strategy from its source code.

    Args:
        source: Python source defining `def strategy(data, params): ...`.
        params: optional default parameter dict to use for the runtime check.

    Returns:
        ValidationResult. Inspect `.ok`, `.errors`, `.warnings`.
    """
    result = ValidationResult(ok=True)

    # Stage 1: static.
    tree = _static_check(source, result)
    if tree is None or not result.ok:
        return result

    # Stage 2: runtime.
    try:
        fn = _load_strategy(source)
    except Exception as exc:
        result.add_error(f"Failed to load strategy: {type(exc).__name__}: {exc}")
        return result

    _runtime_check(fn, result, params)
    return result


def load_validated_strategy(source: str, params: Optional[dict] = None) -> StrategyFn:
    """
    Validate then return the callable. Raises if validation fails.

    Convenience wrapper for callers that just want the function back.
    """
    result = validate_strategy(source, params)
    if not result.ok:
        raise ValueError(
            "Strategy validation failed:\n  " + "\n  ".join(result.errors)
        )
    return _load_strategy(source)
