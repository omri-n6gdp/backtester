"""MACD signal — long when MACD line is above its signal line.

MACD = EMA(fast) - EMA(slow); signal = EMA(MACD, signal_period). Long
when MACD > signal, flat otherwise. A smoothed version of SMA crossover
with less whipsaw at the cost of more lag.
"""
import pandas as pd


def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal_p = int(params.get("signal", 9))
    if fast >= slow:
        raise ValueError("fast must be < slow")

    ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_p, adjust=False).mean()
    return (macd > signal).astype(float)
