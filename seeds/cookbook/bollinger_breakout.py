"""Bollinger breakout — long while close > upper band.

Computes SMA(period) ± k*stdev(period). Goes long when the close
breaks above the upper band and stays long while above the middle
band; flat otherwise. Captures persistent breakouts; vulnerable to
quick fades.
"""
import pandas as pd


def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    k = float(params.get("k", 2.0))

    close = data["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + k * std

    pos = pd.Series(0.0, index=data.index)
    state = 0.0
    for i in range(len(close)):
        c = close.iloc[i]
        u = upper.iloc[i]
        m = mid.iloc[i]
        if pd.isna(u) or pd.isna(m):
            pos.iloc[i] = state
            continue
        if state == 0.0 and c > u:
            state = 1.0
        elif state == 1.0 and c < m:
            state = 0.0
        pos.iloc[i] = state
    return pos
