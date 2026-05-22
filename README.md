# Backtester

A personal-use, fully-free backtesting and deployment engine for trading
strategies. Write a strategy as Python or plain English, test it across
assets and parameter combinations, get deep robustness statistics, rank
every strategy you've ever tried, and deploy the good ones to a
Trading 212 paper account.

Everything runs locally on your own machine. No cloud, no hosting, no
costs, no API bills.

## How it works

Two pieces, both on `localhost`:

- **A local Python server** (`server.py`) — a FastAPI app on
  `127.0.0.1:8000`. It fetches price data, runs the backtests, talks to
  Trading 212, and stores results.
- **A browser frontend** — a single static page the server itself
  serves. It draws the charts and the UI; all the heavy work happens in
  the server.

You start the server once and use the app in your browser.

## Prerequisites

- **Python 3.11 or newer** (this project runs on 3.13).
- A free **Groq API key** — get one at <https://console.groq.com>. Only
  needed for the plain-English → strategy translation feature.
- A **Trading 212 API key + secret** (Settings → API) — only needed for
  the Live and Deploy tabs. Demo account only.

## One-time setup

Open a terminal and run these one at a time:

```bash
cd ~/code/backtester
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then create a file named `.env` in the project root with your keys:

```
GROQ_API_KEY=your_groq_key_here
T212_DEMO_KEY=your_t212_key_here
T212_DEMO_SECRET=your_t212_secret_here
```

The `.env` file stays on your machine and is never committed to git.

## Running it

Every time you want to use the app:

```bash
cd ~/code/backtester
source .venv/bin/activate
python server.py
```

Leave that terminal window open — it's the server. Then open a browser at:

```
http://127.0.0.1:8000/
```

To stop the server, press **Ctrl+C** in its terminal window. **After any
code change, restart the server** (Ctrl+C, then `python server.py` again)
so it picks up the new code.

## Using it

- **Author** — click **+ New** in the sidebar to write a strategy, in
  Python or plain English (the app translates English to code for you).
- **Backtest** — pick a strategy, set the ticker / dates / costs on the
  right, and click **Run Backtest**. The equity chart shows a buy & hold
  baseline so you can see if the strategy actually beat doing nothing.
- **Optimise** — below any backtest, click **Optimise this strategy**. It
  sweeps a grid of parameters, picks the best one (penalised for the
  number of trials so a lucky combo can't fool you), and shows the
  improved curve next to the baseline — plus a regime breakdown and a
  re-test of the winning parameters on other assets.
- **Favourite** — star a strategy; favourites pin to the top of the
  sidebar and the library.
- **Library** — every run is stored and ranked by a composite score you
  can re-weight live.
- **Deploy / Live** — connect a Trading 212 **Demo** account, deploy a
  strategy on a schedule, and watch expected-vs-actual on the Tracking
  tab. Demo (paper) money only.

## Where your data lives

```
~/.backtester/
├── prices/      cached price history (Parquet, per timeframe)
├── strategies/  one Python file per saved strategy
└── results/     results.db — every run, metric, and curve (SQLite)
```

Deleting `~/.backtester/` resets everything; the price cache rebuilds
itself on the next run.

## Backing up your data

Everything that matters lives in `~/.backtester/` — your saved strategies,
your run history, the price cache. The repo itself is replaceable (a `git
clone` rebuilds it); your data is not.

On macOS the easiest backup is **Time Machine**: open System Settings →
General → Time Machine, plug in an external drive, and leave it on. Time
Machine includes your whole home folder (so `~/.backtester/` is covered)
unless you've explicitly excluded it.

If you prefer a one-shot copy, the whole directory is small and portable:

```bash
cp -R ~/.backtester ~/backtester-backup-$(date +%Y-%m-%d)
```

## Running the tests

```bash
cd ~/code/backtester
source .venv/bin/activate
python -m pytest -q
```

## Project layout

```
backend/    the Python server — data, engine, API, deployment
frontend/   the browser app (index.html + app.js + api.js)
tests/      the test suite
scripts/    smoke_backtest.py — an end-to-end sanity run
server.py   start here
CLAUDE.md   full project context, roadmap, and design decisions
```

## A safety note

The Deploy and Live features are wired to a Trading 212 **Demo** account
only — paper money. Real-money (ISA) trading is deliberately gated and
not enabled. Don't change that without understanding exactly what it does.
