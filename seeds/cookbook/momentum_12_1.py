"""12-1 momentum — long when the 12-month return (excluding the most
recent month) is positive.

The classic equity-momentum factor: at each bar, look back ~12 months
and skip the last ~1 month (to avoid short-term reversal noise). Long
when that return is positive, flat otherwise. With daily data we
default to 252-bar lookback skipping the most recent 21.
"""
import pandas as pd


def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    lookback = int(params.get("lookback", 252))
    skip = int(params.get("skip", 21))
    if skip >= lookback:
        raise ValueError("skip must be < lookback")

    close = data["close"]
    # Return from `lookback` bars ago to `skip` bars ago.
    ret_12_1 = (close.shift(skip) / close.shift(lookback)) - 1.0
    return (ret_12_1 > 0).astype(float)
