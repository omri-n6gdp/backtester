# Backtesting Engine — Project Context (CLAUDE.md)

Read this file in full at the start of every session. The "Where we are right now" section at the bottom is the most important part — it tells you the active resume pointer.

## Mission

Build a personal-use, fully-free backtesting and deployment engine for trading strategies. The user supplies a strategy either as a Python function or as a plain-English description; the engine tests it across many assets, timeframes, and parameter combinations; produces deep statistical analysis of robustness and consistency; ranks every strategy ever tested; and can deploy strategies to a Trading 212 paper or live account. Performance, realism, and overfitting-resistance are non-negotiable.

---

## Architecture: Path B — Browser frontend + local Python helper

The system has two pieces, both running on the user's own machine:

**Frontend.** A single static HTML/JS app that renders the UI, charts, configuration, and results. Lives in the browser. Does no heavy computation itself — it sends requests to the local helper and renders responses.

**Local Python helper.** A FastAPI server bound to `localhost:8000`. Handles every operation the browser cannot: yfinance data fetching, Trading 212 API calls (CORS prohibits direct browser calls), Python strategy execution, vectorized backtesting math, on-disk Parquet caching, and SQLite result storage. Started once with `python server.py`; runs in the background while the user uses the frontend.

**Why this split.** Yahoo Finance and Trading 212 both refuse browser-origin requests. Python strategy execution and high-throughput parameter sweeps need NumPy/pandas, which are far faster natively than via Pyodide. Local Parquet caching is the fastest possible data path. Keeping the heavy work server-side also keeps API keys (Trading 212, Groq) out of the browser, so the frontend can stay shareable.

**What the user runs.** One terminal command (`python server.py`) and then opens `http://127.0.0.1:8000/` in a browser. Both stay on `localhost`. No cloud, no hosting, no costs.

---

## Working with Claude Code (CRITICAL — read every session)

This section is the operational contract between Omri and Claude Code. Every session must respect it. Earlier in the project this section described a setup-script-based workflow used with claude.ai chat; that workflow is now deprecated. Claude Code edits files in this repo directly, runs commands itself, and commits to git. The conventions below reflect that.

### About Omri

Omri is **not a software engineer.** He understands the project deeply but doesn't read or write code fluently. He runs commands as a careful non-developer would: copy-paste from prompts, expect things to work, paste error output back when they don't. Claude Code must never assume Omri can read a Python traceback and figure out what to do — read the traceback yourself, identify the bug, and ship a fix.

This means:
- **Step-by-step prose, not "you'll need to add this somewhere."** When Claude Code needs to explain what's happening, every action is one numbered step with copy-pasteable commands.
- **No "just edit the file" by hand.** Omri does not open files in editors during normal work. All edits are done by Claude Code directly via its file-edit tools.
- **No assumed context between commands.** If something depends on a value from an earlier output, say "this is the strategy_id we just generated; we'll use it next."
- **Plain English over jargon.** "FastAPI app factory" → "the file that creates the web server." Explanations go into asides labeled as such, not into the action steps.

### How code arrives now (Claude Code mode)

- Claude Code reads relevant files into context with the `view` tool **before** writing or modifying anything. Don't guess at engine signatures — read them.
- File edits happen via `str_replace`, `create_file`, or full rewrites. There are no more `setup_*.sh` heredoc scripts.
- Tests are run by Claude Code itself with `pytest` over the bash tool. Smoke runs are launched the same way.
- When tests pass, Claude Code commits with a clear message. Each "task" in a phase should be one focused commit, not a megacommit at the end.
- Before declaring a phase done, run the explicit "Done When" checkpoint in the roadmap and show the output to Omri.

### Permissions and oversight

Claude Code will ask Omri before running anything destructive or anything that talks to the outside world for the first time. Omri should approve actions thoughtfully but not anxiously — file edits and pytest runs are routine; git pushes, deletions, and any code path that hits Trading 212 (Phase 10+) deserve careful review.

### Historical pattern: setup scripts (deprecated, but you'll see remnants)

Phases 0–5.7 were built via shell scripts using single-quoted heredocs (`<<'EOF'`), delivered via `present_files` in claude.ai chat, and run by Omri. The repo's `.gitignore` covers `setup_*.sh` and `fix*.sh` patterns from that era. There's no harm in those patterns existing; just don't write new ones unless there's a reason. The current workflow is direct file edits.

### Working in two terminals

Most work needs **two terminals**: one running `python server.py` (the server window — leave it alone), one for everything else (claude code, curl, git, pytest). When Claude Code instructs Omri to "open a second terminal," it means literally a new Terminal window (⌘N on macOS) or tab (⌘T in iTerm), and the new terminal needs `cd ~/code/backtester && source .venv/bin/activate` before it can run pytest etc.

### After any code change, restart the server

The server loads code into memory on startup. Editing `routes.py` after the server is running has no effect until the server is restarted. The reminder: in the server terminal, **Ctrl+C** to stop, then `python server.py` to relaunch.

### Building JSON payloads for `curl`

When testing routes from the shell, inline JSON in a `curl -d "..."` argument with backslash-escaped quotes is fragile. Use a heredoc to a file, then `curl -d @file.json` instead. Every multi-line POST in this project uses that pattern.

### Errors I won't make again (lessons across phases)

These are concrete bugs that cost real time. They're listed so future Claude Code sessions don't repeat them:

- **Don't guess at engine signatures.** Phase 3 took three round-trips of test failures because signatures in `validator.py`, `metrics.py`, and `vectorized.py` were assumed. Always `view` the relevant files first.
- **`load_validated_strategy` returns just the function** (and raises `ValueError` on failure). It does NOT return a tuple. To check validity without raising, use `validate_strategy(code)` which returns a `ValidationResult` with `.ok`, `.errors`, `.warnings`.
- **`compute_metrics(result)` takes one argument.** No `bars_per_year` keyword. The function determines it internally.
- **`get_prices` takes `datetime.date` objects, not ISO strings.** API routes must parse `req.start` / `req.end` with `date.fromisoformat()` before passing them through. There's a `_parse_date()` helper in `backend/api/routes.py`.
- **The FastAPI startup hook fires unreliably under TestClient** in some Starlette versions. Test fixtures call `migrate()` directly after monkeypatching the storage paths, instead of relying on the hook.
- **`backend.api.routes` imports `STRATEGIES_DIR` and `get_prices` by name at module load.** Tests that monkeypatch `backend.storage.STRATEGIES_DIR` or `backend.data.prices.get_prices` must ALSO monkeypatch `backend.api.routes.STRATEGIES_DIR` and `backend.api.routes.get_prices`, or routes keep using the original references.
- **Scripts in `scripts/` add the project root to `sys.path`** at the top using `Path(__file__).resolve().parent.parent`. No editable install needed.
- **Don't open frontend HTML via `file://`.** Browsers refuse to treat `null`-origin pages as valid origins for ES module imports. Always serve `frontend/` over HTTP via the same FastAPI process; mount `StaticFiles(directory="frontend", html=True)` AFTER `create_app()` so API routes still win on path resolution.
- **lightweight-charts v5 changed the addSeries API.** Use `chart.addSeries(LWC.AreaSeries, opts)`, NOT v4's `chart.addAreaSeries(opts)`. Most blog posts still show the v4 API and silently produce a blank chart.
- **lightweight-charts requires strictly increasing time values.** When converting a tz-aware pandas DatetimeIndex to seconds-since-epoch, dedupe in the loop.
- **Sandbox-test setup scripts against real engine code, not just AST-parse.** AST parsing only catches syntax errors. From Phase 5 onward: when tests are part of the deliverable, run pytest before declaring done.
- **`total_return` is not multiplicatively composable across the equity-curve start point.** Use raw equity factors (`equity[-1] / starting_capital`) when you need a two-factor identity like `IS × OOS == Full`.
- **First-bar-zero artifact in walk-forward concatenation.** Each test window's first bar contributes zero return. For N windows that's N missed bars. Consistent with "deploy fresh each window" semantics; documented, not a bug.
- **Calendar-aware test failures look like new bugs but aren't.** `pd.date_range(end=datetime.now(), freq='B')` snaps to the prior business day on weekends. Failures on a Saturday + numbers-off-by-one is a strong signal the bug is calendar-aware. Use fixed start dates in fixtures.
- **Don't strip NaN combos silently in walk-forward selection.** Filter explicitly; if every combo NaNs, raise a clean `ValueError` rather than silently picking an arbitrary index.
- **Cross-module coupling on leading-underscore names is a maintenance trap.** Phase 5.7 reimplemented `expand_grid` / `_select_best` in `sweep.py` with identical semantics to walk-forward's versions, accepting ~50 lines of duplication for clean boundaries. Lift to a shared util only when a third caller appears.
- **DSR's V parameter is best computed empirically when N trials are known.** The conservative default `V = 1/T` assumes IID Gaussian under H0; with an actual sweep, `Var(per-bar trial Sharpes)` is a stronger estimator. `compute_deflated_sharpe` accepts `trial_sharpes_variance` as an optional kwarg for exactly this.
- **Failed combos in a sweep should be visible artifacts, not silently dropped.** They surface as an `error` field and zeroed metrics. Dropping them would lie about the parameter surface (heatmaps especially).
- **Per-combo robustness analyses (OOS, MC, regimes) are deliberately NOT run per combo in sweeps.** Would 5x runtime. Reserved for the best-combo only via the existing single-backtest route.
- **T212 v0 uses HTTP Basic with `(api_key, secret)`, NOT a bare API key.** The "Settings → API (Beta)" page shows two values — both halves are required. `httpx.BasicAuth(key, secret)` is the standard way; manual base64 isn't needed. Probing with bare-key auth 401s on every endpoint (including the no-scope `metadata/instruments`), looking identical to "key revoked" — don't waste time regenerating until the auth shape is verified. The first 43-char value alone is the *secret* (it's longer than the key); having only that, with no api-key partner, produces uniform 401s.
- **T212 enforces aggressive per-endpoint rate limits** (account/cash ~1/2s, portfolio/orders ~1/5s, pies ~1/30s). The browser's refresh button plus lazy-load on tab activation will trigger 429s on back-to-back identical calls. Phase 10.5 added a per-endpoint TTL cache in `backend/api/routes.py` (`_t212_cached`) tuned above T212's limits; exceptions are NEVER cached, so failures don't poison the entry. New T212 routes should route through this cache (or have their own — like `/t212/instruments`'s 24h Parquet cache).
- **T212 instrument tickers are not the same as the human shortName.** `ticker` is e.g. `AAPL_US_EQ` (US-listed) or `AAPLd_EQ` (Deutsche-listed); `shortName` is `AAPL`. A user-supplied "AAPL" can map to multiple T212 venues — `find_by_ticker()` returns a list, not a single match. Suffix conventions: `_US_EQ` US equity, lowercase `d_EQ` Deutsche listing.

---

## Tech stack

**Backend (Python 3.11+; the project is currently running on 3.13.9):**
- `fastapi` + `uvicorn` — HTTP server
- `yfinance` — primary data source. Note: yfinance is an unofficial scraper and Yahoo aggressively rate-limits it. The data layer includes throttling, exponential backoff on 429s, browser-fingerprint impersonation, and an optional fallback source.
- `curl_cffi` — pulled in as a hard dependency by yfinance ≥ 0.2.60. yfinance now **rejects stock `requests.Session` objects** and requires a `curl_cffi.Session`, which impersonates real browser TLS/JA3/HTTP2 fingerprints. We rotate the impersonate target (chrome124, chrome131, chrome133a, safari17_0, safari18_0, edge101) per call — strictly stronger than User-Agent rotation alone for evading 429s.
- `pandas`, `numpy` — vectorized backtest math
- `pyarrow` — Parquet read/write for the price cache
- `numba` (optional) — JIT compilation; add only if profiling shows it's needed
- `httpx` — Trading 212 API calls and Stooq fallback fetch
- `groq` — natural-language → strategy code translation (model: `llama-3.3-70b-versatile`)
- `python-dotenv` — load API keys from `.env`
- `sqlite3` (stdlib) — results database
- `pytest`, `pytest-asyncio` — unit tests
- `matplotlib` — dev convenience only, used by the smoke script for the equity-curve PNG. Not part of the production runtime.

**Frontend:**
- React via `esm.sh` + import maps — gives component reactivity without a build step. Single `index.html` is still the deliverable.
- `lightweight-charts` v5 for price/equity charts
- `tailwindcss` via CDN for styling
- `codemirror` via CDN for the in-browser strategy editor (Phase 6+)
- No `localStorage`; results live in the SQLite database via the API

**No npm install, no webpack, no Docker.** The whole thing should clone-and-run.

---

## Data sources (all free)

**Primary: yfinance.** Covers stocks, ETFs, indices, FX pairs, crypto (BTC-USD, ETH-USD, etc.), and commodities via ETFs (GLD for gold, USO for oil, etc.). Daily, hourly, and minute data with varying lookback. Free, no key, used server-side only. **Rate-limit hardening is mandatory** — see Tech Stack note above.

**Crypto-native fallback: CoinGecko + Binance public endpoints.** Used when finer crypto granularity than yfinance offers is needed.

**General fallback for resilience: Stooq.** Free CSV endpoints for daily equity/index data; backup when yfinance is throttled. **Daily-only.**

**News (optional, Phase 7+): GDELT** (free, unlimited) over NewsAPI (free tier is 100/day and doesn't allow commercial use).

**LLM: Groq.** Free tier (no credit card; 30 RPM / 6,000 TPM / 1,000 requests per day on most models). Used for natural-language → strategy translation. Default model: `llama-3.3-70b-versatile`.

---

## Caching strategy

Every fetched dataset is stored on disk as Parquet. On each run, the engine fetches only the gap between the last cached timestamp and now ("incremental update"). A 20-year daily history of 100 assets is a few hundred MB. Hourly and minute data are cached separately per timeframe.

**Layout:**
```
~/.backtester/
├── prices/
│   ├── 1d/AAPL.parquet, BTC-USD.parquet, ...
│   ├── 1h/...
│   ├── 5m/...
│   └── meta.json          # last-fetched timestamps per (asset, timeframe)
├── strategies/
│   └── {strategy_id}.py   # one Python file per strategy.
└── results/
    └── results.db         # SQLite — strategies, runs, metrics, curves, trades

# .env lives in the project repo root, NOT in ~/.backtester/.
```

---

## Strategy authoring

The user can author strategies two ways:

**1. Python code.** A strategy is a single pure function with this signature:

```python
def strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    data:   OHLCV DataFrame indexed by timestamp (UTC, tz-aware).
            Columns: open, high, low, close, volume.
    params: dict of strategy parameters.
    returns: pd.Series of target position sizes aligned with data.index.
             Convention: -1.0 (full short) ... 0.0 (flat) ... +1.0 (full long).
             Fractional values allowed for sizing (e.g. 0.5 = half long).
             Values outside [-1, +1] are rejected by the validator.
             Engine shifts by 1 bar internally to prevent look-ahead.
    """
```

**2. Plain English (Phase 6).** The user types a description ("Buy when 50-day MA crosses above 200-day MA, sell when it crosses below, hold long only"). The frontend sends this to the helper, which calls Groq with a system prompt that constrains output to the strategy function signature above.

**Validation step.** Before any strategy is saved or run at scale, it goes through a two-stage validator (`backend/engine/validator.py`):
1. **Static (AST):** rejects banned imports (only `pandas`, `numpy`, `math`, `datetime` allowed), banned calls (`eval`, `exec`, `compile`, `open`, `__import__`, `input`, `exit`, `quit`), and dunder attribute access (`__class__`, `__globals__`, …).
2. **Runtime:** loads the function in a constrained namespace, runs it on a deterministic synthetic OHLCV sample (252 days, seeded), asserts the output is a numeric `pd.Series` of the right length with values in `[-1, +1]` after dropping NaN.

**Validator API surface:**
- `validate_strategy(source: str, params: Optional[dict] = None) -> ValidationResult` — does both stages, never raises. Returns a result object with `.ok`, `.errors`, `.warnings`, `.strategy_name`. Use for clean error surfacing to the user (400 responses).
- `load_validated_strategy(source: str, params: Optional[dict] = None) -> StrategyFn` — convenience wrapper that calls `validate_strategy` and either returns the callable or raises `ValueError`. Use when invalid input is exceptional (re-validating stored code at backtest time).

---

## Backtesting engine

**Vectorized core (default, fast).** Used for all parameter sweeps. The math:

```
raw_positions = strategy(data, params)            # target on bar t
positions     = raw_positions.shift(1).fillna(0)  # LOOK-AHEAD GUARD: applied at bar t+1
returns       = data['close'].pct_change()        # bar returns
costs         = compute_costs(positions, data, cost_config)
strat_ret     = positions * returns - costs
equity        = (1 + strat_ret).cumprod() * starting_capital
```

The `BacktestResult` dataclass exposes both `raw_positions` (what the strategy returned) and `positions` (the shifted ones actually applied), plus `equity`, `drawdown`, `returns`, `costs`, and a list of round-trip `Trade` objects.

**Engine API surface:**
- `run_backtest(strategy, data, params, cost_config=..., starting_capital=..., bars_per_year=...) -> BacktestResult`
- `compute_metrics(result) -> MetricsBundle` — takes one positional argument, no kwargs.

**Event-driven mode (Phase 9, optional).** For strategies that need stop-losses, take-profits, or intra-bar limit fills. Iterates bar by bar. Used only when the user opts in.

**Realistic-execution toggles** (each independently on/off; see `backend/engine/costs.py`):
- **Slippage** — fixed bps (default 5) or volatility-scaled (`'vol_scaled'` mode uses 0.5 × 14-bar ATR%)
- **Commission** — bps of notional (default 0; T212 is commission-free)
- **Spread** — applied separately from slippage
- **FX fee** — Trading 212 charges 0.15% on non-GBP instruments; default 15 bps; auto-zeroed when `instrument_is_gbp=True`
- **Borrow cost** — annualized bps for short positions, applied per-bar proportional to `abs(position)` only when negative
- **Min lot size** — Phase 9 / event-driven
- **Next-bar fill** — built in via the mandatory `.shift(1)`. Cannot be disabled.

---

## Metrics computed for every backtest

Implemented in `backend/engine/metrics.py`. One function per metric, all pure, all unit-tested.

**Return:** total return, CAGR, best/worst month.
**Risk:** annualized volatility, max drawdown, max DD duration (bars→days), avg drawdown, Ulcer Index, downside deviation, VaR (95%, 99%), CVaR (95%).
**Risk-adjusted:** Sharpe, Sortino, Calmar, Omega, MAR.
**Trade-level:** number of trades, win rate, profit factor, expectancy, avg win, avg loss, avg holding period, max consecutive wins/losses.
**Benchmark (Phase 7):** alpha, beta, correlation, information ratio vs SPY.
**Curves stored:** equity, drawdown, per-bar returns/costs, raw + shifted positions.

---

## Robustness analysis (anti-overfitting layer) — Phase 5

This is what separates a real backtester from a curve-fitting toy. Every strategy run gets:

**In-sample / out-of-sample split (5.1).** User specifies a split date (default 70/30). All optimization happens on in-sample only. OOS reported alongside but never used for parameter selection.

**Walk-forward optimization (5.2).** Rolling training window (e.g. 3y) → optimize → test on next window (e.g. 1y) → roll forward. Reported performance is the concatenation of all out-of-sample windows. Gold standard for honest parameter selection.

**Monte Carlo trade-order shuffling (5.3).** Randomly reorder trade returns 1000 times, recompute equity. Output: P5/P50/P95 equity band, distribution of max DD, distribution of CAGR. Tells the user how lucky vs. skilled the strategy was.

**Deflated Sharpe (5.4).** Bailey & López de Prado's deflated Sharpe ratio. Parameter sweeps are penalized for the number of trials run.

**Regime conditional analysis (5.5).** Performance broken down by: bull market (SPY 200-day rising), bear market (SPY 200-day falling), high-vol (VIX > 25), low-vol (VIX < 15).

**Parameter heatmaps (5.6).** For 2D parameter sweeps, render a heatmap of any chosen metric.

**Sweep route (5.7).** Consolidates the grid loop, DSR with empirical V, and heatmaps into one endpoint. Used by Phase 5.8's robustness dashboard.

---

## Strategy ranking system — Phase 8

Every backtest run is stored permanently in `results.db`. The frontend has a "Strategy Library" view that ranks all strategies by a configurable composite score. Default weights: 30% OOS Sharpe / 20% Calmar / 15% consistency / 15% deflated Sharpe / 10% breadth / 10% recency. User can edit weights live and the ranking re-sorts.

---

## Trading 212 deployment

T212 has an official beta API with Demo and Live environments. Auth: Basic with `key:secret` base64-encoded. Available for Invest and Stocks ISA accounts (CFD is not supported). Market, Limit, Stop, Stop-Limit orders supported. The API is **still not idempotent in beta** — client-side deduplication via request IDs is mandatory.

**Safeguards:** kill switch in the UI; per-strategy max position size; daily loss circuit breaker; explicit "I understand" dialog before any live order; idempotency wrapper around T212 calls.

---

## Setup checklist (high-level)

1. Install Python 3.11 or later (project runs on 3.13.9).
2. `pip install fastapi uvicorn yfinance pandas numpy pyarrow httpx groq python-dotenv pytest pytest-asyncio` (numba added later only if needed).
3. Generate Trading 212 API keys (Settings → API) — Demo only until Phase 13.
4. Get a free Groq API key at console.groq.com.
5. Clone the repo, run `python server.py`, open `http://127.0.0.1:8000/` in a browser.

---

## Build Roadmap — Zero to Complete

Authoritative ordering. Each phase has a **Done When** checkpoint — don't move on until that condition is true. Commit to git after every task that meaningfully changes the codebase.

---

### Phase 0 — Prerequisites and one-time setup ✅ COMPLETE

Python 3.13.9 venv at `~/code/backtester/.venv`. Project skeleton: `backend/`, `frontend/`, `tests/`, `scripts/`. `~/.backtester/` cache root with `prices/{1d,1h,5m}/`, `strategies/`, `results/`. `.env` in project root with `GROQ_API_KEY` and `T212_DEMO_KEY` populated. `.gitignore` covers `.env`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `results.db`, `.backtester/`, `.DS_Store`, `setup_*.sh`, `fix*.sh`, `smoke_equity.png`. `backend/config.py` exposes frozen `settings` dataclass.

---

### Phase 1 — Data layer ✅ COMPLETE

`backend/data/yfinance_source.py` with throttle, backoff, curl_cffi impersonation rotation. `backend/data/stooq_source.py` fallback. `backend/data/prices.py` with `get_prices(ticker, timeframe, start, end)` (date objects, not strings). Parquet cache with atomic writes, gap-fill, meta.json. 11 tests green.

**Baseline:** SPY 2010→today, ~4100 daily bars, ~210 KB Parquet.

---

### Phase 2 — Backtest core ✅ COMPLETE

`contract.py`, `validator.py` (two-stage AST + runtime), `costs.py`, `vectorized.py` (look-ahead guard), `metrics.py` (24+ metrics). 18 engine tests + 11 prices = 29 green. `scripts/smoke_backtest.py` end-to-end.

**Smoke baseline (drifts slightly day-to-day as tail extends):** SPY 2010→today, SMA(50/200), CAGR ~9.04%, Sharpe ~0.69, Max DD ~−33.7%, 8 trades, 75% win rate, $410k final equity from $100k.

---

### Phase 3 — FastAPI backend ✅ COMPLETE

`backend/api/__init__.py` factory. Routes: `/health`, `/strategies` (POST/GET), `/strategies/{id}`, `/backtest`. `backend/db/` (schema, conn, repo). `server.py` binds to `127.0.0.1:8000`. 40 tests green.

---

### Phase 4 — Minimal frontend ✅ COMPLETE

`frontend/index.html`, `api.js`, `app.js` served by FastAPI `StaticFiles` at `http://127.0.0.1:8000/`. Three-panel layout. Lightweight-charts v5 equity curve. Buy & Hold comparison deferred to Phase 7.

---

### Phase 5 — Robustness analyses

**5.1 OOS split ✅** `backend/engine/robustness/oos.py`. `split_result(result, split=0.7)` returns IS/OOS `WindowResult`s with rebased equity. Boundary trades dropped. Wired into `/backtest` via optional `oos_split` field. 19 tests; keystone `test_compound_consistency_is_times_oos_equals_full` asserts math integrity.

**5.2 Walk-forward ✅** `backend/engine/robustness/walk_forward.py`. `walk_forward(strategy, data, param_grid, train_bars, test_bars, ...)`. Grid is dict (cartesian) or list-of-dicts. Returns per-window `WindowReport`s, concatenated OOS series, aggregate metrics, `param_stability`. New route `POST /backtest/walk_forward`. Walk-forward runs persist with `params_json: {mode: "walk_forward", ...}`. Trades NOT persisted (window-relative). 28 tests; keystone `test_train_window_does_not_see_test_bars` empirically proves no look-ahead.

Smoke verdict on SMA crossover with 9-combo grid, 12 windows on SPY 2010–2026: aggregate OOS Sharpe **0.68** (vs single 0.69), CAGR **5.2%** (vs 9.0%), max DD **−21%**. Param stability ~0. **SMA crossover is real but its parameters are not learnable.**

**5.3 Monte Carlo trade-order shuffling ✅** `backend/engine/robustness/monte_carlo.py`. 1000 shuffles of trade returns; output P5/P50/P95 equity bands, DD distribution, CAGR distribution. Key invariance: `final_equity` is permutation-invariant; only max DD and DD duration are order-dependent. 19 tests. Wired into `/backtest` via optional `monte_carlo` config.

Smoke on SPY SMA: all 500 final_equities = $410,298.40 (invariance confirmed); original max DD −8.36% sat at 74.6th percentile of shuffles; q05 = −14.60%.

**5.4 Deflated Sharpe ✅** `backend/engine/robustness/deflated_sharpe.py`. Implements Bailey & López de Prado's PSR and DSR using stdlib `statistics.NormalDist` (no scipy). 23 tests + 5 new tests in `test_deflated_sharpe_variance.py` (Phase 5.7).

Smoke on SPY SMA with `n_trials=9`: sharpe=0.689, skew=−0.557, excess_kurt=25.17 (fat tails), E[max SR|H0,N=9]=0.377, PSR=0.997, DSR=0.893. The ~10pp PSR→DSR gap is the selection-bias tax for 9 trials.

**5.5 Regime conditional analysis ✅** `backend/engine/robustness/regimes.py`. Two axes: `spy_trend` (bull/bear by SPY vs 200-day SMA) and `vix_level` (low_vol <15 / normal_vol 15-25 / high_vol >25). Route fetches SPY (400-day warmup) and `^VIX` via existing `get_prices`; per-axis fetch errors isolated. 21 tests.

Smoke verdict: SPY bull 83.9% (CAGR +18.7%, Sharpe 1.51); bear 16.1% (CAGR −29.8%, all 8 trades entered in bull — bear-CAGR reflects existing-position drag during slow exits). VIX low_vol 35% (Sharpe 4.99); normal_vol 51% (Sharpe 0.28); high_vol 13.5% (Sharpe −1.71). Edge concentrates in low-VIX bars; SMA crossover's bull/bear partition is partially tautological since its entry rule IS an SMA cross.

**5.6 Parameter heatmaps ✅** `backend/engine/robustness/heatmaps.py`. Pure data-shaping module — given `[(params_dict, metrics)]`, builds a 2D grid of one metric across two named param axes. Five aggregations (mean/median/min/max/last) for duplicate cells. NaN/missing_mask twin output (NaN cells in `grid`, True in `missing_mask` — boolean mask exists because NaN isn't valid JSON). 30 tests. No schemas/routes patches — pure data shaping called by 5.7.

**5.7 Sweep route ✅** `backend/engine/robustness/sweep.py` + new `POST /backtest/sweep` route. Consolidates the grid loop, deflated Sharpe (with empirical V from per-trial Sharpes), and heatmaps. Per-combo metrics returned for every combo; the heavy `BacktestResult` is kept only on the winner (memory hygiene). Failed combos surface as visible artifacts (`error` field + zeroed metrics). Per-combo OOS/MC/regimes are OFF by default — would 5x runtime. `compute_deflated_sharpe` extended with optional `trial_sharpes_variance` kwarg (backward-compat). Sweep runs persist with `params_json: {mode: "sweep", ...}`. 27 sweep tests + 5 new DSR variance tests. **212 tests passing total.**

**5.8 Frontend: robustness dashboard ✅** `frontend/app.js` got an Author/Run view split AND mode-aware right rail: Backtest mode exposes OOS/DSR(n_trials)/MC/regimes toggles; Sweep mode has a param-grid JSON textarea + selection metric + heatmap axes + DSR toggle. New components: `DSRCard` (PSR/DSR/expected-max-SR/skew/kurtosis), `OOSCompare` (IS vs OOS cards side by side, OOS highlighted), `MCCard` (actual max DD + percentile rank + P5/P50/P95 DD quantiles), `RegimePanel` (spy_trend + vix_level tables), `HeatmapGrid` (pure-CSS color-mapped grid with legend), `ComboTable` (sorted by score, best combo starred). MC equity bands are trade-indexed (length = n_trades+1) so they cannot be overlaid on the time-series equity chart — the dashboard shows MC distributional stats as a stat card instead, which is what the data actually represents. `SweepResponse` gained `best_combo_timestamps` + `best_combo_equity` so the sweep canvas can render the winner's equity curve. Smoke verified on SPY SMA(9-combo grid): 9/9 heatmap fill, DSR n_trials=9 visible, best combo identified visually. PSR→DSR gap is small (0.6pp) because empirical V is small on a flat parameter surface — that's correct behaviour, not a bug.

**5.9 Test consolidation ✅** `tests/conftest.py` extracted: `api_client` fixture (tmp dirs + storage monkeypatches + migrate + TestClient), `make_synthetic_ohlcv`, `install_fake_prices`, `post_strategy` helpers. Four API test files (`test_api`, `test_api_oos`, `test_api_walk_forward`, `test_sweep`) lost ~25 lines of duplicated setup each (net −219 lines of boilerplate, +91 lines of shared infra). New `tests/test_api_robustness.py` (+6 tests) closes the gap on route wiring for `monte_carlo`, `n_trials`, `include_regimes` — engine math was unit-tested but the routes never were. Also fixed a redundant `object.__setattr__` artifact in `_make_combo` (test_sweep.py).

**Done When (5.8 / 5.9):** Running the SMA crossover with a sweep produces a heatmap, OOS metrics, walk-forward equity curve, MC bands, and a deflated Sharpe in the UI. The deflated Sharpe is meaningfully lower than the raw best-of-grid Sharpe.

---

### Phase 6 — Strategy authoring UI ✅ COMPLETE

`backend/llm/groq_client.py` exposes `translate_strategy(description: str) -> str`. System prompt encodes the strategy contract (signature, allowed imports, banned constructs, position convention) plus 3 few-shot examples: SMA crossover, RSI mean reversion, momentum. Model: `llama-3.3-70b-versatile`. `_get_client()` is a `@lru_cache` seam that tests monkeypatch to a fake — zero real Groq calls in the suite. Markdown fence stripping is defensive: LLMs ignore "code only" instructions sometimes.

Two new routes: `POST /strategies/validate` (dry-run validation, no DB write, always 200) and `POST /strategies/translate` (Groq → code + pre-validation result; 400 if description is empty, 502 on Groq failure).

Frontend: CodeMirror 6 via esm.sh import map (`codemirror`, `@codemirror/lang-python`, `@codemirror/theme-one-dark`). New top-level `view: 'run' | 'author'`; "+ New" button in sidebar opens `AuthorView`. Author UI: name/description inputs, plain-English textarea + Generate button, CodeMirror editor (Python highlighting, dark theme), Validate + Save buttons. `ValidationPanel` component shows pass/fail with inline errors and warnings. Save pre-validates, POSTs `/strategies`, refreshes sidebar, auto-selects the new strategy, returns to Run mode. Right rail hidden in author mode.

Tests (+13): `tests/test_llm.py` (6 tests) and `tests/test_api_authoring.py` (7 tests). **231 tests passing total.**

**Done When ✅ verified end-to-end via live server:** translate "Buy when RSI(14) crosses above 30, sell when above 70" → validation passed → save "RSI Mean Reversion" → backtest SPY 2020-2024 → 11 trades, CAGR 2.77%, Sharpe 0.243.

---

### Phase 7 — Full run configurator and results dashboard ✅ COMPLETE

Frontend: cost-config editor in the right rail (slippage mode + bps, commission, spread, FX with GBP toggle, borrow — wired to `BacktestRequest.cost_config` / `SweepRequest.cost_config`). Backtest canvas gained `DrawdownChart` (red AreaSeries under equity), `MonthlyHeatmap` (pure-CSS years×months grid, computed client-side from per-bar returns, annual-total column), `TradeList` (client-side sortable table with show-all toggle), `PinStrip` + `ComparisonView` (in-memory pins up to 3; `MultiEquityChart` overlays as colored LineSeries with a side-by-side metric table).

Schema fix: `CostConfigModel` previously used `fx_fee_bps` / `borrow_cost_bps`, but the engine dataclass uses `fx_bps` / `borrow_bps_annual`. The mismatch was latent — no Phase 0-6 code path actually sent a cost_config. Aligned the schema to the engine names; also added `bars_per_year` for full parity. No backend tests changed.

Scope decision: sweep-combo pinning was deferred. Currently only the best combo carries `full_result` (others are pruned for memory). Phase 8's library view is the better place to pin runs from any source.

**Done When ✅** verified: SPY SMA(50/200) with slippage 5 vs 30 bps produces CAGR 8.30% vs 8.05% (cost wired). Drawdown / monthly heatmap / trade list all render. Pinning 2 runs overlays them in the comparison view.

---

### Phase 8 — Strategy library and ranking ✅ COMPLETE

`backend/engine/ranking.py` — `compute_run_scores(rows, breadth_map, now)` returns 6 normalized component scores per run in [0,1]: sharpe, calmar, consistency (= sortino/sharpe proxy), breadth (log-saturating around 5 tickers), recency (exp-decay, 90-day half-life), and dsr (null — see below). No internal weighting; the frontend combines.

`backend/db/repo.py` got the read side it always needed: `get_run`, `get_metrics`, `get_curves`, `get_trades`, `list_runs_with_metrics(strategy_id?, ticker?)`, `count_distinct_tickers_per_strategy`.

New routes: `GET /library?strategy_id=&ticker=` (joined runs + metrics + component scores; filters AND-ed) and `GET /runs/{run_id}` (returns the same shape as `POST /backtest`; sweep / walk-forward runs have empty trades — deliberate from earlier phases). `BacktestResponse` gained `strategy_id` so the library click-through can sync the sidebar selection.

Frontend: `LibraryView` with 6 live weight sliders (CLAUDE.md defaults 30/20/15/15/10/10), ticker filter, sortable table with sweep / walk_forward mode badges. Row click → `getRun(id)` → switch to backtest canvas with that run loaded.

Honest substitution notes (call this out if a user asks about scoring):
- **DSR isn't persisted.** `dsr_score` is null on every row; the frontend redistributes its weight to available components per row, so DSR-less runs aren't unfairly zeroed.
- **"OOS Sharpe" lands on raw Sharpe.** OOS metrics aren't persisted today (the route computes them on request and returns them in the response, but only the full-period `metrics_json` reaches the DB). The CLAUDE.md doc-default 30% weight for "OOS Sharpe" is honored as raw Sharpe weight.
- **Consistency** is a sortino/sharpe ratio scaled to [0,1] — a gain-stability proxy, not true rolling-Sharpe stability. Getting the real thing requires loading curves per ranked run; deferred until it matters.

Adding persisted DSR / OOS for true scoring is a schema change (extra columns or sidecar tables) — a future cleanup phase.

Tests: 11 ranking unit tests + 5 API tests. **247 tests passing total.**

**Done When ✅** verified: 8 stored runs from prior sessions return ranked with all 6 component scores. `GET /runs/{id}` round-trips with 3522 bars and 7 trades. Ticker filter works. 404 on bad id.

---

### Phase 9 — Event-driven backtest mode ✅ COMPLETE

`backend/engine/event_driven.py` — `StopConfig` dataclass (optional `stop_loss_pct` / `take_profit_pct` / `trailing_stop_pct`, all non-negative fractions in `[0, 1)`) plus `run_event_driven(strategy, data, params, *, stop_config, cost_config, starting_capital, bars_per_year) -> BacktestResult`. Same shape as vectorized; the full downstream stack (metrics, OOS, MC, regimes, library) works unchanged.

Engine algorithm:
- Vectorize signals (same `.shift(1)` look-ahead guard as vectorized).
- Walk bar-by-bar updating `peak_high` / `trough_low` for trailing stops.
- Check stops in priority order **stop_loss → take_profit → trailing**.
- On trigger: fill at the trigger price, but if the bar gapped past it (open below SL on a long), fill at the worse open. Partial bar return: held from prev_close to stop_price, then flat.
- Re-entry: stop closes the position mid-bar (`actual_positions[i] = 0`); if the signal still wants the position, we re-open at THIS bar's close so the NEXT bar holds the new position. "No immediate re-entry" applies only WITHIN the stop-fire bar.
- Float epsilon `_PRICE_EPS = 1e-9` on trigger comparisons — prevents `100.0 * 1.10 = 110.00000000000001` from silently suppressing at-threshold stops. Real bug caught during testing.

`Trade.exit_reason: str | None` added to the dataclass — None for vectorized trades, `'stop_loss' / 'take_profit' / 'trailing_stop' / 'signal'` for event-driven. Backwards-compatible.

`POST /backtest/event_driven` route; persists `params_json: {mode: 'event_driven', stop_config: {...}}` so the library can badge it. Optional analyses (OOS / MC / DSR / regimes) deliberately NOT wired into this route — they're computed at request time on `/backtest` and `on_bar`-style live state hooks were flagged optional in this phase.

Frontend: "Event-driven (intra-bar stops)" toggle in the Backtest right rail; three percent inputs (SL / TP / trailing); routes to `/backtest/event_driven` when toggled. `TradeList` shows a "Why exit" column when any trade has a non-null `exit_reason` (vectorized runs unchanged). Library badge `stops` (purple) for `event_driven` runs.

Tests (+13): `tests/test_event_driven.py` (10) and `tests/test_api_event_driven.py` (3). **260 tests passing total.**

**Done When ✅ verified live:** SPY 2010-2024 long-and-hold + 5% trailing stop produced **137 trailing-stop exits** (138 trades, last is the final-bar close). Exit prices near peak × 0.95 as expected. CAGR 3.94% vs vectorized 12.89% — trailing stops correctly chop a steady uptrend into pieces.

**Scope notes carried forward:**
- The `on_bar(state, bar) -> orders` strategy form is still deferred (CLAUDE.md flagged it optional). Adding it would be a small new module without disturbing the existing one.
- Limit/stop entry orders are skipped — entries are at the bar's close after a signal change, same as vectorized.
- Numba JIT skipped (CLAUDE.md says only if profiling warrants); the loop runs in ms-scale on daily data.

---

### Phase 10 — Trading 212: read-only integration

`backend/t212/client.py` wraps httpx with Basic auth. Methods: `get_account_cash`, `get_positions`, `get_instruments`, `get_orders`, `get_pies`. `backend/t212/instruments.py` maps tickers to T212 instrument codes, cached daily. Routes: `GET /t212/account|positions|instruments`. Frontend "Live" view shows Demo account.

**Done When:** Live view's Demo tab shows real T212 demo account cash, positions, P&L.

---

### Phase 11 — Trading 212 deployment (demo only)

New SQLite tables: `deployments`, `orders_sent`, `fills_received`. `backend/deploy/runner.py` computes target position deltas. `backend/deploy/orders.py` builds idempotent orders (request ID = hash of `deployment_id + as_of_timestamp + side + qty`). `backend/deploy/scheduler.py` (APScheduler acceptable). Live vs backtest tracking comparison.

**Done When:** Deployment of SMA crossover on AAPL to Demo runs daily for 2+ days, places correct orders, Tracking tab plots live vs expected.

---

### Phase 12 — Safeguards

Kill switch (flatten all). Per-deployment max position. Daily loss circuit breaker. Confirmation modals ("I understand"). Each safeguard fails closed.

---

### Phase 13 — Going live (ISA)

Generate `T212_LIVE_KEY`. Per-deployment "live activation token" flow. Smoke test read-only on Live. First live deployment with tiny sizing (£50 cap, -2% circuit breaker). Compare live tracking to Demo. Scale up only after 2+ weeks of expected behavior.

---

### Phase 14 — Polish

Structured logging to `~/.backtester/logs/`. Error toast UI. Cache compaction maintenance script. Time Machine backups. README + STRATEGIES.md + DEPLOYMENT.md. Replace deprecated `@app.on_event("startup")` with modern `lifespan`.

---

## Phase ordering rationale

- **0 → 1 → 2** ✅ got to a working backtest with no UI fast. ✅
- **3 → 4** ✅ thin slice end-to-end. ✅
- **5 (robustness) before 6 (authoring)** — intentional. Adding strategies before honest grading produces a library full of overfit junk.
- **6 → 7 → 8** is the productivity loop: author → configure → analyze → rank.
- **9 (event-driven) is optional** and can be deferred.
- **10 → 11 → 12 → 13** is the deployment path; each step strictly subordinate. Demo only until safeguards work; live only after Demo is boring.
- **14 (polish) is continuous** but the listed items must be in place before trusting the system to run unattended.

---

## Out of scope (explicit non-goals)

- Multi-user / cloud / hosting. Personal-use tool; everything local.
- Tick-level data. Daily / hourly / minute is sufficient.
- Options, futures, leverage beyond T212. Equities, ETFs, and crypto only.
- Real-time streaming quotes. Trading happens at end-of-bar.
- Order types beyond what T212 supports.

---

## Conventions for Claude Code on this project

These conventions are binding.

### Operational
- **Default to Path B's architecture.** Browser ↔ FastAPI on localhost.
- **Never recommend paid APIs or paid services.** Free tier or nothing.
- **The roadmap is the contract.** "Do the next thing" means look up the next unchecked task in the Build Roadmap. Don't skip.
- **Always source the venv before pytest.** Omri's prompt shows `(base)` (conda) by default; the project's pytest, fastapi, etc. live in `.venv`. Prefix Python-using commands with `source .venv/bin/activate` whenever the prompt isn't already `(.venv)`.

### Engineering correctness
- **Vectorized first.** Only fall back to event-driven when the feature genuinely requires bar-by-bar state.
- **No look-ahead bias, ever.** Every signal shifted by ≥1 bar before being applied. The `.shift(1).fillna(0.0)` line in `backend/engine/vectorized.py:run_backtest` is the project's keystone invariant.
- **Test relative effects, not absolute thresholds, for noisy quantities.** Absolute Sharpe thresholds at finite N are seed-dependent; relative reductions are seed-robust.
- **Anti-overfitting first.** When the user runs a sweep, the engine should automatically include walk-forward / OOS / deflated-Sharpe results. Don't make it opt-in.
- **Realistic costs on by default.** Default slippage 5 bps, default commission 0, default FX fee 15 bps for non-GBP.
- **Cache, don't refetch.** Check Parquet cache first; fetch only the missing tail.
- **yfinance requires curl_cffi sessions.** Stock `requests.Session` is rejected. Use `curl_cffi.requests.Session(impersonate=...)` and rotate impersonate targets. Close sessions in a `finally` block.

### Security & secrets
- **Confirm before live trading.** Any code path that hits `live.trading212.com` requires explicit confirmation token from the frontend. Demo is the default.
- **Keep secrets out of the frontend.** API keys live in `.env` server-side only.
- **Minimal dependencies.** If stdlib + pandas + numpy can do it, don't add a library.
- **Scripts in `scripts/` add the project root to `sys.path`** at the top (using `Path(__file__).resolve().parent.parent`) so `python scripts/foo.py` works without an editable install.

### Working with Claude Code specifically
- **Read before writing.** `view` the engine files (`vectorized.py`, `metrics.py`, `contract.py`, etc.) before adding code that imports from them. Phase 3 cost real time because signatures were assumed.
- **Sandbox-test before declaring done.** Run pytest in the bash tool before committing. Don't trust AST-parse alone.
- **One focused commit per task.** Not megacommits at the end of phases.
- **Run the "Done When" checkpoint and show Omri the output.** Don't claim done without proof.
- **Ask before destructive ops, network calls outside known surfaces, or anything touching money.** Routine file edits and pytest don't need permission.

---

## Where we are right now (resume pointer)

**Last completed:** Phase 11.7 — deployment frontend (Deploy + Tracking views). **414 backend tests passing, all green.** Phase 11 is PARTIAL: 11.1–11.7 are built; the **11.8 Done-When checkpoint** (deploy to Demo, run 2+ days, verify) has NOT been done yet.

Phase 10 (read-only Demo integration) shipped earlier and remains green: `backend/t212/client.py` (httpx + `BasicAuth`, read methods + error hierarchy), `backend/t212/instruments.py` (24h Parquet cache + `find_by_ticker`), `GET /t212/*` routes with a TTL cache, and the frontend `LiveView`. See "Commits" / git log for detail.

State at the end of Phase 11.4:
- `backend/t212/client.py` — gained write methods: `place_limit_order`, `place_market_order`, `cancel_order`, `get_order`. T212 v0 order endpoints live-probed; full 15-field response schema documented in the docstrings. Two probe gotchas captured in "Errors I won't make again": DELETE has a post-place race window; trailing slash on `/equity/orders/{id}/` 404s.
- `backend/db/schema.sql` — three new tables: `deployments` (strategy + ticker + capital + cron + `armed` flag), `orders_sent` (UNIQUE `request_id_hash` is the idempotency key), `fills_received`. `backend/db/repo.py` has the matching insert/get/list/update helpers.
- `backend/deploy/runner.py` — `compute_intents(...)`: pure-compute. Strategy signal → target $-value → target qty (round 4dp) → diff vs current T212 portfolio → list of `OrderIntent`. NaN signal → flat (mirrors backtest `.fillna(0.0)`); dust threshold 1e-4 shares. No DB writes, no T212 calls.
- `backend/deploy/orders.py` — `submit_intents(conn, deployment_id, intents)`: hashes each intent (`compute_request_id_hash`), inserts into `orders_sent` (duplicate hash → `SKIPPED_DUPLICATE`), calls the T212 client, records `SUBMITTED` / `FAILED`. Caller owns the transaction. FAILED is terminal for that hash; next bar's fresh `as_of` retries.

**No live order has been placed by the deploy engine yet** — 11.1's probe placed + cancelled one manual test order; the runner/submitter/scheduler have only been exercised against mocked T212 in tests.

**Commits in Phase 11 so far:** `05cca03` 11.1 client writes (8 tests) · `8212a58` 11.2 schema + repo (14 tests) · `addc6ac` 11.3 runner (22 tests) · `bc11192` 11.4 idempotent submitter (18 tests) · `96efbc4` 11.5 scheduler (19 tests) · `a1bef8e` 11.6 routes (22 tests) · `b1648ee` 11.7 frontend.

**Next up:** **Phase 11.8 — Done-When checkpoint** (deploy SMA crossover on AAPL to Demo, run 2+ days, verify orders placed + Tracking plots expected-vs-actual).

Remaining Phase 11 scope:
- **11.5** ✅ `backend/deploy/scheduler.py` — APScheduler `BackgroundScheduler`, one cron job per armed deployment (job id `deployment:<id>`, in-memory jobstore, `register_all_armed()` rebuilds jobs from SQLite on boot). **Public API for 11.6 to wire routes to:** `run_deployment(id) -> DeploymentRunResult` (full pipeline, never raises; statuses OK / DISARMED / NOT_FOUND / NO_DATA / ERROR), `register_deployment(deployment_dict)`, `unregister_deployment(id)`, `register_all_armed()`, `start()`, `shutdown()`, `get_scheduler()`. `as_of` = last bar's timestamp, so re-firing the same trading day is idempotent. `armed` checked at registration AND fire time. Misfires are skipped, not run late. `resolve_price_ticker` bridges the T212 instrument code → yfinance symbol (the one place the two identifiers meet). Wired into `server.py` startup/shutdown hooks (kept out of `create_app()` so the test suite never spawns a thread). 19 tests.
- **11.6** ✅ Seven `/deployments` routes in `backend/api/routes.py` + 8 schemas in `schemas.py`. `POST /deployments` (requires `confirmation_token: "ARM"`; creates the deployment **armed** and registers its cron job), `GET /deployments` (filters: strategy_id, armed_only), `GET /deployments/{id}`, `POST /deployments/{id}/arm` (also requires the ARM token), `POST /deployments/{id}/disarm` (no token — turning off is frictionless), `POST /deployments/{id}/run` (manual trigger; 200 + status OK/DISARMED/NO_DATA/ERROR), `GET /deployments/{id}/tracking` (deployment + `orders_sent` + a best-effort "expected" backtest curve that degrades to `expected_error` if it can't run). **ARM model:** every transition that ENABLES live orders (create, arm) needs the token; disarm and a manual run of an already-armed deployment don't. 22 tests.
- **11.7** ✅ Frontend (`frontend/app.js` + `api.js`). New top-bar **Deploy** button → `DeployView`: a create-deployment form (strategy, T212 ticker, timeframe, capital, order type, cron, params JSON) + a deployments list with per-row Arm / Pause / Run now / Tracking. `ArmModal` gates create + arm (the user types ARM); pause is one click. `DeploymentTrackingPanel`: config summary + expected equity curve (the strategy backtested) + expected CAGR/Sharpe/maxDD + the orders-placed table. `api.js` got the 7 `/deployments` calls. No backend change; the frontend has no test runner — `node --check` only. **Browser verification still pending** — and a running server must be restarted to pick up 11.5–11.7.
- **11.8** Done-When: deploy SMA crossover on AAPL to Demo, run 2+ days, verify orders placed + Tracking plots live-vs-expected.

**Decisions locked for the rest of Phase 11:** APScheduler for scheduling; `confirmation_token: "ARM"` required on `POST /deployments` even for Demo (reused with stricter wording for Phase 13 live).

**Persistent strategy ID for smoke runs:** `896013b850a24a1b8fbc47bee509c671` (SPY SMA crossover; persisted from Phase 5.3).

**Schema gap to remember (still relevant for any future scoring/ranking work):** OOS / DSR / MC results aren't persisted to SQLite today — the route computes them on request and ships them in the response, but only the full-period `metrics_json` reaches the DB. The library's ranking therefore uses raw Sharpe in place of OOS Sharpe and nulls DSR. Fixing this is a schema change (extra columns or sidecar tables).

**To resume in a fresh Claude Code session:**
1. `cd ~/code/backtester`
2. `claude` (or `claude resume` to continue a prior session)
3. First prompt: "Read CLAUDE.md fully. Confirm by summarizing the current resume pointer and the next phase. Don't write any code yet — let's align before starting."
4. After confirmation: "Let's do the Phase 11.8 Done-When checkpoint."

The full repo state is reconstructable from git (`git log --oneline` on `main`); the price cache lives at `~/.backtester/prices/`; the T212 instruments cache at `~/.backtester/t212/`; `.env` has the required `T212_DEMO_KEY` + `T212_DEMO_SECRET`. Total tests: **414 passing**; smoke baseline on SPY 2010–2026 SMA(50/200) sits at CAGR ~9.05%, Sharpe ~0.69, Max DD ~−33.7%, 8 trades.
