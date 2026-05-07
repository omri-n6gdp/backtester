PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS strategies (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    code_path   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
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
