# Writing strategies

A strategy answers one question, bar by bar: **how much of this asset do
I want to hold right now?** You can write it two ways — plain English or
Python.

## The plain-English way

In the app, click **+ New**, type a description into the English box, and
click **Generate**. For example:

> Buy when the 50-day moving average crosses above the 200-day moving
> average, sell when it crosses back below. Long only.

The app sends that to an LLM, which writes the Python for you. The code
appears in the editor — review it, click **Validate**, then **Save**.

## The Python way

A strategy is one function with exactly this shape:

```python
def strategy(data, params):
    ...
    return positions
```

- **`data`** — a table of prices, one row per bar, with the columns
  `open`, `high`, `low`, `close`, `volume`. It's a pandas DataFrame
  indexed by timestamp.
- **`params`** — a dictionary of settings (e.g. `{"short": 50, "long": 200}`).
  Always read them with a default: `params.get('short', 50)`.
- **return value** — a pandas Series, one number per bar, that says how
  much to hold:
  - `1.0` = fully long, `0.0` = flat (in cash), `-1.0` = fully short.
  - Fractions are allowed: `0.5` = half long.
  - Values outside `-1.0 … +1.0` are rejected.

### Example: moving-average crossover

```python
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 50))
    long = int(params.get('long', 200))
    sma_short = data['close'].rolling(short).mean()
    sma_long = data['close'].rolling(long).mean()
    # Hold long whenever the fast average is above the slow one.
    return (sma_short > sma_long).astype(float)
```

### Example: RSI mean reversion

```python
import pandas as pd

def strategy(data, params):
    period = int(params.get('period', 14))
    delta = data['close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    # Go long when the asset is oversold; sit flat otherwise.
    return (rsi < 30).astype(float)
```

## The rules

Strategies run inside a sandbox. Before one is saved or run, a validator
checks it:

- **Allowed imports:** only `pandas`, `numpy`, `math`, `datetime`.
- **Banned:** `eval`, `exec`, `compile`, `open`, `__import__`, `input`,
  `exit`, `quit`, and reaching into dunder attributes (`__class__`,
  `__globals__`, …). These are blocked so a strategy can't escape the
  sandbox or touch your machine.
- The function must return a numeric Series the same length as `data`,
  with every value in `-1.0 … +1.0`.

## No look-ahead — automatically

You never have to worry about accidentally "trading on tomorrow's price."
The engine shifts every signal forward by one bar before applying it: a
position your strategy decides on at the close of bar *t* is held
starting at bar *t+1*. This guard cannot be turned off — it's the single
most important rule in the whole project.

## Parameters and Optimise

Anything you read from `params` is a knob you can tune. Run a backtest,
then click **Optimise** and give it a grid like:

```json
{"short": [10, 20, 50], "long": [100, 150, 200]}
```

The app backtests every combination, picks the best one (penalised for
how many it tried, so a lucky fluke can't win), and shows the improved
curve next to your baseline — plus a regime breakdown and a re-test of
the winning parameters on other assets.
