# Deploying a strategy to Trading 212

Deployment lets a strategy run on a schedule and place orders on a
Trading 212 account by itself. **Everything here is Demo (paper money)
only.** Real-money trading is deliberately gated and not enabled.

## Before you start

1. In Trading 212, open **Settings → API (Beta)** and generate a key for
   the **Demo** (Practice) account. You get *two* values — an API key
   and a secret. You need both.
2. Put them in the project's `.env` file:
   ```
   T212_DEMO_KEY=your_key_here
   T212_DEMO_SECRET=your_secret_here
   ```
3. Restart the server so it picks them up.

## Check the connection first

Open the **Live** tab. If the keys work you'll see your Demo account's
cash, positions, and P&L. If you see errors, the keys are wrong or the
account type isn't supported (Invest / Stocks ISA only — CFD is not).

## Creating a deployment

Open the **Deploy** tab and fill in the form:

- **Strategy** — which saved strategy to run.
- **Ticker** — the Trading 212 instrument code, *not* the plain symbol.
  Apple on the US exchange is `AAPL_US_EQ`, for example. The same company
  can exist on several exchanges, so the code includes the venue.
- **Timeframe**, **capital**, **order type** (market or limit).
- **Schedule** — a 5-field cron expression, interpreted in UTC. For a
  daily run a few minutes after the US close, something like
  `5 21 * * 1-5` (21:05 UTC, weekdays).
- **Params** — the strategy parameters, as JSON.

When you create or arm a deployment you must type **ARM** into the
confirmation box. That word is the "I understand this will place orders"
gate — turning a deployment *off* needs no confirmation.

## Running and watching it

Each deployment row has:

- **Arm / Pause** — arm puts it live on its cron schedule; pause stops it
  and cancels any still-working order.
- **Run now** — trigger one run immediately, instead of waiting for the
  cron time. Re-running on the same trading day is safe — orders are
  de-duplicated, so you can't accidentally place a second one.
- **Tracking** — expected vs actual: the strategy backtested over the
  deployment window, the orders actually placed, and their fills.
- **Delete** — removes the deployment and cancels its working orders.

If the laptop was asleep at the scheduled minute, the missed run is
caught up automatically the next time the server starts.

## The safety net

Every safeguard fails closed — when in doubt, it places nothing:

- **Max position size** — a hard cap on how large a position a deployment
  may build, independent of its capital.
- **Circuit breaker** — set a loss limit; if the open position falls
  past it, the deployment disarms itself and stops trading.
- **Kill switch** — a prominent control that disarms *every* deployment
  and places closing orders for all open positions at once. It asks you
  to confirm before it fires.
- **Idempotency** — one order per deployment per bar per side, even if a
  run fires twice.

## What's logged

Every scheduled run, catch-up, FX conversion, and error is written to
`~/.backtester/logs/backtester.log` (rotating). If a deployment behaves
unexpectedly, that file is the place to look.

## Going live (real money)

It is not enabled, on purpose. Live trading is a separate, gated phase
that needs a deliberate decision, a separate live API key, and tiny
starting sizes. Don't switch it on without understanding exactly what
each safeguard does and watching the Demo account behave correctly for
a couple of weeks first.
