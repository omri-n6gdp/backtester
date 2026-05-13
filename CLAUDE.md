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

**5.8 Frontend: robustness dashboard — IN PROGRESS / NEXT.** When a sweep run completes, the main canvas should show: IS vs OOS metrics side by side; walk-forward concatenated equity curve; MC confidence band overlaid on the equity curve; regime metrics table; parameter heatmap (rendered with d3 or a lightweight grid).

**5.9 Test consolidation.** Final cleanup pass across all 5.x modules.

**Done When (5.8 / 5.9):** Running the SMA crossover with a sweep produces a heatmap, OOS metrics, walk-forward equity curve, MC bands, and a deflated Sharpe in the UI. The deflated Sharpe is meaningfully lower than the raw best-of-grid Sharpe.

---

### Phase 6 — Strategy authoring UI

Author and save new strategies in the browser, both as Python code and from plain English.

CodeMirror 6 via CDN with Python syntax highlighting. "Validate" button → `POST /strategies/validate` (new dry-run-only route). "Save" → `POST /strategies`. `backend/llm/groq_client.py` with `translate_strategy(description: str) -> str`. System prompt constrains output to the strategy function signature; few-shot examples (SMA crossover, RSI mean reversion, momentum). `POST /strategies/translate` returns generated code without saving.

**Done When:** Typing "Buy when RSI(14) crosses above 30, sell when above 70" → Generate → review → Validate (passes) → Save → it appears in the sidebar and runs successfully.

---

### Phase 7 — Full run configurator and results dashboard

Polished right-rail configurator: asset/timeframe/date pickers, parameter sweep UI, realistic-execution toggles, robustness checkboxes. Main canvas: equity + drawdown + monthly heatmap + trade list + full metrics table + comparison/pinning mode. Phase 7 also indexes columns on the `trades` table for sortable queries.

**Done When:** Configure → run → see all panels in one scrollable view. Pinning two runs from a sweep shows them overlaid.

---

### Phase 8 — Strategy library and ranking

`GET /library?sort=...&filter=...`. `backend/engine/ranking.py` with the default composite weights. Sortable/filterable table with click-through to results. Live weight editor (client-side re-sort).

**Done When:** After running 10+ strategies/sweeps across 5+ assets, library shows them all with sensible composite scores; adjusting weights re-orders them live.

---

### Phase 9 — Event-driven backtest mode (optional)

`backend/engine/event_driven.py`. Bar-by-bar; supports market/limit/stop orders; stop-loss, take-profit, trailing stop. Optional `on_bar(state, bar) -> orders` strategy form. Numba JIT only if profiling shows it's needed.

**Done When:** Strategy with a 5% trailing stop runs and the trade list shows stop-hit exits at expected prices.

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

**Last completed:** Phase 5.7 — sweep route consolidating robustness analyses. **212 tests passing, all green.**

State at the end of Phase 5.7:
- `backend/engine/robustness/sweep.py` — `run_sweep(strategy, data, param_grid, *, base_params=None, ..., selection_metric="sharpe", higher_is_better=True, heatmap_requests=None, include_deflated_sharpe=True) -> SweepResult`. `SweepResult` exposes: combos, n_combinations, selection_metric, higher_is_better, best_index, best_combo, deflated_sharpe, heatmaps, param_keys. `ComboResult` has: index, params, metrics, score, error, full_result (only on winner).
- `backend/engine/robustness/deflated_sharpe.py` — extended with optional `trial_sharpes_variance: float | None = None` kwarg. When provided, replaces the conservative `V = 1/T` default; the sweep route supplies the empirical variance of per-trial Sharpes for a stronger DSR.
- `backend/api/schemas.py` — added `HeatmapRequestModel`, `HeatmapResponseModel`, `ComboResultModel`, `SweepRequest`, `SweepResponse` (inserted before TradeModel anchor).
- `backend/api/routes.py` — new `POST /backtest/sweep` endpoint. Marshal helpers `_combo_to_model` and `_heatmap_to_model` inserted before `# -------- /health --------`. Sweep persists to `runs` with `params_json: {mode: "sweep", ...}` so the Phase 8 library view can distinguish them.

**Next up:** **Phase 5.8 — frontend robustness dashboard.** This is the visual payoff phase. The backend exposes everything; now the dashboard needs to make it intelligible.

Suggested 5.8 scope (Claude Code can refine):
1. New right-rail "Robustness" config block: OOS toggle, walk-forward toggle, Monte Carlo toggle (with `n_shuffles` input), regimes toggle, heatmap axis pickers.
2. When a single-backtest run includes OOS/MC/regime/DSR results, render: IS vs OOS metrics side-by-side card; MC P5/P50/P95 band overlaid on equity curve; regime metrics table.
3. New "Sweep" run mode: configure the param grid + heatmap axes → call `POST /backtest/sweep` → render the heatmap (with a numeric/colormap legend); show the per-combo table sorted by selection metric; render best-combo equity curve.
4. Subtle UI surfacing for DSR: a small badge near the headline Sharpe showing PSR/DSR; tooltip explains the gap.

**Smoke test target for 5.8:** SPY SMA crossover, `param_grid: {"short": [10, 20, 50], "long": [100, 150, 200]}`, with `heatmaps: [{"x_param": "short", "y_param": "long", "metric_name": "sharpe"}]`. Expected: 9 combos rendered, heatmap fills 9/9, DSR with `n_trials=9` shown, best combo identified visually.

**Persistent strategy ID for smoke runs:** `896013b850a24a1b8fbc47bee509c671` (SPY SMA crossover; persisted from Phase 5.3).

**To resume in a fresh Claude Code session:**
1. `cd ~/code/backtester`
2. `claude` (or `claude resume` to continue a prior session)
3. First prompt: "Read CLAUDE.md fully. Confirm by summarizing the current resume pointer and the next phase. Don't write any code yet — let's align before starting."
4. After confirmation: "Let's do Phase 5.8."

The full repo state is reconstructable from git (`git log --oneline` on `main`); the cache lives at `~/.backtester/`; `.env` has the required keys. Total tests: **212 passing**; smoke baseline on SPY 2010–2026 SMA(50/200) sits at CAGR ~9.05%, Sharpe ~0.69, Max DD ~−33.7%, 8 trades.
