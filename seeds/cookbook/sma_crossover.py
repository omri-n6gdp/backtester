"""SMA crossover — long when a fast moving average is above a slow one.

Classic trend-following starter strategy. Long when SMA(short) > SMA(long),
flat otherwise. Reacts slowly so it pays a lot in whipsaw trades but
captures the bulk of long bull runs.
"""
import pandas as pd


def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    short = int(params.get("short", 50))
    long = int(params.get("long", 200))
    if short >= long:
        raise ValueError("short must be < long")
    fast = data["close"].rolling(short).mean()
    slow = data["close"].rolling(long).mean()
    return (fast > slow).astype(float)
