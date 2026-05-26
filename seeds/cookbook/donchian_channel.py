"""Donchian channel — long on N-bar high, exit on M-bar low.

Buy the breakout, sell the breakdown. Buy when today's close is the
highest of the last N bars; sell when it's the lowest of the last M.
A staple of trend-following CTAs.
"""
import pandas as pd


def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    entry_n = int(params.get("entry_n", 20))
    exit_m = int(params.get("exit_m", 10))

    close = data["close"]
    rolling_high = close.rolling(entry_n).max()
    rolling_low = close.rolling(exit_m).min()

    pos = pd.Series(0.0, index=data.index)
    state = 0.0
    for i in range(len(close)):
        c = close.iloc[i]
        h = rolling_high.iloc[i]
        l = rolling_low.iloc[i]
        if pd.isna(h) or pd.isna(l):
            pos.iloc[i] = state
            continue
        if state == 0.0 and c >= h:
            state = 1.0
        elif state == 1.0 and c <= l:
            state = 0.0
        pos.iloc[i] = state
    return pos
