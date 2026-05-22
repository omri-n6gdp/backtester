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
