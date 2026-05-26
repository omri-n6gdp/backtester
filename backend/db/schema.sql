PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS strategies (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    code_path   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1,
    favourite   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(active);

-- Phase 16.2: content-hashed strategy versions. Every save (create
-- or update) inserts a row when the hash is new; the most recent
-- row is the strategy's "current" code. Runs link to the exact hash
-- they ran against (runs.strategy_code_hash) so editing a strategy
-- does not silently invalidate prior backtest results.
CREATE TABLE IF NOT EXISTS strategy_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    code_hash    TEXT NOT NULL,
    code         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE,
    UNIQUE (strategy_id, code_hash)
);
CREATE INDEX IF NOT EXISTS idx_strategy_versions_strategy
    ON strategy_versions(strategy_id, created_at DESC);

CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    strategy_id      TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    start_date       TEXT NOT NULL,
    end_date         TEXT NOT NULL,
    params_json      TEXT NOT NULL,
    cost_config_json TEXT NOT NULL,
    starting_capital REAL NOT NULL,
    bars_per_year    INTEGER NOT NULL,
    created_at       TEXT NOT NULL,
    -- Phase 16.1: OOS / DSR / Monte Carlo results persisted alongside
    -- the run so the library ranking can read real OOS Sharpe + real
    -- DSR (instead of falling back to full-period Sharpe / null).
    -- Nullable: old runs and analyses that weren't requested stay null.
    oos_json         TEXT,
    dsr_json         TEXT,
    monte_carlo_json TEXT,
    -- Phase 16.2: links a run to the exact strategy_versions row that
    -- ran. Nullable for pre-16.2 runs; populated by every new backtest.
    strategy_code_hash TEXT,
    -- Phase 16.3: user-supplied annotations. `notes` is free text; `tags`
    -- is a normalized comma-separated list (lowercase, alphanumeric +
    -- hyphen, no spaces). Both nullable.
    notes              TEXT,
    tags               TEXT,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_id);
CREATE INDEX IF NOT EXISTS idx_runs_created  ON runs(created_at);

CREATE TABLE IF NOT EXISTS metrics (
    run_id       TEXT PRIMARY KEY,
    metrics_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS curves (
    run_id              TEXT PRIMARY KEY,
    timestamps_json     TEXT NOT NULL,
    equity_json         TEXT NOT NULL,
    drawdown_json       TEXT NOT NULL,
    returns_json        TEXT NOT NULL,
    costs_json          TEXT NOT NULL,
    raw_positions_json  TEXT NOT NULL,
    positions_json      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS trades (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);

-- Phase 11: Trading 212 deployment ---------------------------------------
-- `deployments` is the configuration: which strategy on which T212 ticker,
-- with what budget and schedule. `armed=1` is the gate the scheduler checks.
CREATE TABLE IF NOT EXISTS deployments (
    id                  TEXT PRIMARY KEY,
    strategy_id         TEXT NOT NULL,
    ticker              TEXT NOT NULL,            -- T212 instrument code, e.g. "AAPL_US_EQ"
    timeframe           TEXT NOT NULL,
    params_json         TEXT NOT NULL,
    deployment_capital  REAL NOT NULL,            -- account-currency budget for sizing
    cost_config_json    TEXT NOT NULL,
    order_type          TEXT NOT NULL DEFAULT 'market',  -- 'market' | 'limit'
    schedule_cron       TEXT NOT NULL,            -- 5-field crontab, e.g. "5 21 * * 1-5"
    armed               INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    last_run_at         TEXT,                     -- ISO 8601; NULL until first run
    max_position_value  REAL,                     -- Phase 12.3: hard cap on position value; NULL = no cap
    max_loss_value      REAL,                     -- Phase 12.4: circuit breaker on open position loss; NULL = off
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);
CREATE INDEX IF NOT EXISTS idx_deployments_strategy ON deployments(strategy_id);
CREATE INDEX IF NOT EXISTS idx_deployments_armed    ON deployments(armed);

-- `orders_sent` is our local record of every order intent for a deployment.
-- `request_id_hash` is the idempotency key (sha256 of deployment_id +
-- as_of + side + qty + ticker, truncated to 16 hex chars). Duplicate
-- insert raises sqlite3.IntegrityError — the order builder catches that
-- and treats it as "already submitted; skip."
CREATE TABLE IF NOT EXISTS orders_sent (
    id                TEXT PRIMARY KEY,
    deployment_id     TEXT NOT NULL,
    request_id_hash   TEXT NOT NULL UNIQUE,
    t212_order_id     INTEGER,                    -- NULL until T212 accepts
    as_of_timestamp   TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    side              TEXT NOT NULL,              -- 'BUY' | 'SELL'
    qty               REAL NOT NULL,              -- absolute shares
    order_type        TEXT NOT NULL,              -- 'market' | 'limit'
    limit_price       REAL,                       -- NULL for market
    status            TEXT NOT NULL DEFAULT 'PENDING',
    response_json     TEXT,                       -- T212 response or error blob
    created_at        TEXT NOT NULL,
    FOREIGN KEY (deployment_id) REFERENCES deployments(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_orders_deployment ON orders_sent(deployment_id);
CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders_sent(status);
CREATE INDEX IF NOT EXISTS idx_orders_t212id     ON orders_sent(t212_order_id);

-- Phase 14: per-run history — one row per `run_deployment` fire,
-- recording the outcome so the Tracking UI can show a real audit trail
-- (more than just `last_run_at`). Deleting a deployment cascades.
CREATE TABLE IF NOT EXISTS deployment_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id     TEXT NOT NULL,
    ran_at            TEXT NOT NULL,              -- wall-clock ISO 8601 UTC
    status            TEXT NOT NULL,              -- OK / NO_DATA / ERROR / INSUFFICIENT_CASH / CIRCUIT_BREAKER / DISARMED / NOT_FOUND
    as_of_timestamp   TEXT,                       -- bar timestamp the run acted on
    intent_count      INTEGER NOT NULL DEFAULT 0,
    submitted_count   INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    FOREIGN KEY (deployment_id) REFERENCES deployments(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_deployment_runs_dep ON deployment_runs(deployment_id, ran_at DESC);

-- `fills_received` is one row per partial / full fill. T212 typically
-- emits one row per order but the schema allows N to keep the option
-- open without a migration.
CREATE TABLE IF NOT EXISTS fills_received (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id       TEXT NOT NULL,
    t212_order_id  INTEGER,
    filled_qty     REAL NOT NULL,
    filled_price   REAL NOT NULL,
    filled_at      TEXT NOT NULL,
    raw_json       TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders_sent(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills_received(order_id);
