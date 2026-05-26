"""RSI mean reversion — buy oversold, sell overbought.

Wilder's RSI: long when RSI(period) drops below the oversold threshold,
exit when it rises back above the overbought threshold. Works in
range-bound markets, dies in strong trends.
"""
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing: equivalent to an EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))

    rsi = _rsi(data["close"], period)
    pos = pd.Series(0.0, index=data.index)
    state = 0.0
    for i in range(len(rsi)):
        v = rsi.iloc[i]
        if pd.isna(v):
            pos.iloc[i] = state
            continue
        if state == 0.0 and v < oversold:
            state = 1.0
        elif state == 1.0 and v > overbought:
            state = 0.0
        pos.iloc[i] = state
    return pos
