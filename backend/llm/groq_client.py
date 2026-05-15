"""
Groq client for natural-language → strategy code translation.

Single public function: `translate_strategy(description: str) -> str`.
The Groq SDK is constructed on first call via `_get_client()` so tests
can monkeypatch that seam and avoid hitting the network.

The system prompt encodes the strategy contract (signature, allowed
imports, banned constructs, position convention) and three few-shot
examples. Model: llama-3.3-70b-versatile (per CLAUDE.md).
"""
from __future__ import annotations

import re
from functools import lru_cache

from groq import Groq

from backend.config import settings

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You convert plain-English trading strategy descriptions into Python code conforming to this strict contract.

CONTRACT
- Define EXACTLY ONE function: `def strategy(data, params):`
- `data` is a pandas DataFrame indexed by tz-aware UTC timestamps with columns: open, high, low, close, volume.
- `params` is a dict of parameters. Use `params.get('name', default)` for safe access.
- Return a pandas Series aligned with `data.index`. Values MUST be in [-1.0, +1.0]:
    -1.0 = full short, 0.0 = flat, +1.0 = full long. Fractional values allowed.
- ALLOWED IMPORTS (nothing else): pandas, numpy, math, datetime
- BANNED: eval, exec, open, __import__, input, exit, quit; any dunder attribute access (e.g. __class__, __globals__).
- The engine shifts positions by 1 bar internally; DO NOT shift positions yourself.
- Output CODE ONLY. No markdown fences. No prose. No explanation before or after.

EXAMPLES

Description: SMA crossover — buy when fast SMA crosses above slow SMA, sell otherwise.
Code:
import pandas as pd

def strategy(data, params):
    short = int(params.get('short', 50))
    long = int(params.get('long', 200))
    sma_short = data['close'].rolling(short).mean()
    sma_long = data['close'].rolling(long).mean()
    return (sma_short > sma_long).astype(float)

Description: Buy when RSI(14) drops below 30, sell when it rises above 70.
Code:
import pandas as pd
import numpy as np

def strategy(data, params):
    period = int(params.get('period', 14))
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    long_signal = rsi < 30
    flat_signal = rsi > 70
    pos = pd.Series(np.nan, index=data.index)
    pos[long_signal] = 1.0
    pos[flat_signal] = 0.0
    return pos.ffill().fillna(0.0)

Description: Go long when the 20-day momentum is positive.
Code:
import pandas as pd

def strategy(data, params):
    lookback = int(params.get('lookback', 20))
    momentum = data['close'].pct_change(lookback)
    return (momentum > 0).astype(float)
"""


_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown ```python ... ``` fencing if the LLM added one.

    Idempotent and a no-op when there are no fences. We deliberately only
    handle the common single-fence case; if the model emits multiple
    fenced blocks we take whatever's there and let the validator decide.
    """
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    # Fallback: strip a leading ``` line and trailing ``` line if present.
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    """Lazy singleton. Tests monkeypatch this function."""
    return Groq(api_key=settings.groq_api_key)


def translate_strategy(description: str) -> str:
    """Translate a natural-language strategy description to Python code.

    Args:
        description: free-form English describing the strategy.

    Returns:
        Python source as a string. Not validated — caller should run
        `validate_strategy()` on the result before saving.

    Raises:
        ValueError: if `description` is empty.
        Any exception from the Groq SDK (auth, rate-limit, network) is
        allowed to propagate. The API route converts it to a 502.
    """
    if not description or not description.strip():
        raise ValueError("description must not be empty")

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description.strip()},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    return _strip_fences(raw)
