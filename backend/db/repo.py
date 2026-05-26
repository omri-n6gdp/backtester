"""
Data access layer. Pure functions over a sqlite3 connection.

Repo functions never open or close connections — they receive an open one
from the caller (typically a `with connect() as conn:` block in a route).
This keeps transactions composable: a route can do multiple inserts
atomically.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any


# -------- strategies --------

def insert_strategy(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    name: str,
    description: str,
    code_path: str,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO strategies (id, name, description, code_path, created_at, active)
        VALUES (?, ?, ?, ?, ?, 1)
        """,
        (strategy_id, name, description, code_path, created_at),
    )


def list_strategies(conn: sqlite3.Connection, *, include_inactive: bool = False) -> list[dict[str, Any]]:
    # Favourites sort to the top; newest-first within each group.
    if include_inactive:
        rows = conn.execute(
            "SELECT id, name, description, code_path, created_at, active, favourite "
            "FROM strategies ORDER BY favourite DESC, created_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, description, code_path, created_at, active, favourite "
            "FROM strategies WHERE active = 1 ORDER BY favourite DESC, created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_strategy(conn: sqlite3.Connection, strategy_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, name, description, code_path, created_at, active, favourite "
        "FROM strategies WHERE id = ?",
        (strategy_id,),
    ).fetchone()
    return dict(row) if row else None


def soft_delete_strategy(conn: sqlite3.Connection, strategy_id: str) -> bool:
    cur = conn.execute(
        "UPDATE strategies SET active = 0 WHERE id = ? AND active = 1",
        (strategy_id,),
    )
    return cur.rowcount > 0


def set_strategy_favourite(
    conn: sqlite3.Connection, strategy_id: str, favourite: bool
) -> bool:
    """Set a strategy's favourite flag. Returns False when no active
    strategy has that id (soft-deleted strategies cannot be favourited)."""
    cur = conn.execute(
        "UPDATE strategies SET favourite = ? WHERE id = ? AND active = 1",
        (1 if favourite else 0, strategy_id),
    )
    return cur.rowcount > 0


# -------- runs / metrics / curves / trades --------

def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    strategy_id: str,
    ticker: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    params_json: str,
    cost_config_json: str,
    starting_capital: float,
    bars_per_year: int,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            id, strategy_id, ticker, timeframe, start_date, end_date,
            params_json, cost_config_json, starting_capital, bars_per_year, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, strategy_id, ticker, timeframe, start_date, end_date,
            params_json, cost_config_json, starting_capital, bars_per_year, created_at,
        ),
    )


def insert_metrics(conn: sqlite3.Connection, *, run_id: str, metrics_json: str) -> None:
    conn.execute(
        "INSERT INTO metrics (run_id, metrics_json) VALUES (?, ?)",
        (run_id, metrics_json),
    )


def insert_curves(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    timestamps_json: str,
    equity_json: str,
    drawdown_json: str,
    returns_json: str,
    costs_json: str,
    raw_positions_json: str,
    positions_json: str,
) -> None:
    conn.execute(
        """
        INSERT INTO curves (
            run_id, timestamps_json, equity_json, drawdown_json,
            returns_json, costs_json, raw_positions_json, positions_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, timestamps_json, equity_json, drawdown_json,
            returns_json, costs_json, raw_positions_json, positions_json,
        ),
    )


def insert_trades(conn: sqlite3.Connection, *, run_id: str, trades: list[dict[str, Any]]) -> None:
    if not trades:
        return
    conn.executemany(
        "INSERT INTO trades (run_id, seq, data_json) VALUES (?, ?, ?)",
        [(run_id, i, json.dumps(t, default=str)) for i, t in enumerate(trades)],
    )


# -------- reads (Phase 8) --------

def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, strategy_id, ticker, timeframe, start_date, end_date,
               params_json, cost_config_json, starting_capital, bars_per_year, created_at
        FROM runs WHERE id = ?
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def get_metrics(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT metrics_json FROM metrics WHERE run_id = ?", (run_id,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["metrics_json"])


def get_curves(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT timestamps_json, equity_json, drawdown_json, returns_json,
               costs_json, raw_positions_json, positions_json
        FROM curves WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "timestamps": json.loads(row["timestamps_json"]),
        "equity": json.loads(row["equity_json"]),
        "drawdown": json.loads(row["drawdown_json"]),
        "returns": json.loads(row["returns_json"]),
        "costs": json.loads(row["costs_json"]),
        "raw_positions": json.loads(row["raw_positions_json"]),
        "positions": json.loads(row["positions_json"]),
    }


def get_trades(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT data_json FROM trades WHERE run_id = ? ORDER BY seq", (run_id,),
    ).fetchall()
    return [json.loads(r["data_json"]) for r in rows]


# -------- deployments / orders / fills (Phase 11) --------

def insert_deployment(
    conn: sqlite3.Connection,
    *,
    deployment_id: str,
    strategy_id: str,
    ticker: str,
    timeframe: str,
    params_json: str,
    deployment_capital: float,
    cost_config_json: str,
    order_type: str,
    schedule_cron: str,
    armed: bool,
    created_at: str,
    max_position_value: float | None = None,
    max_loss_value: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO deployments (
            id, strategy_id, ticker, timeframe, params_json,
            deployment_capital, cost_config_json, order_type,
            schedule_cron, armed, created_at, max_position_value, max_loss_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            deployment_id, strategy_id, ticker, timeframe, params_json,
            deployment_capital, cost_config_json, order_type,
            schedule_cron, 1 if armed else 0, created_at, max_position_value,
            max_loss_value,
        ),
    )


def get_deployment(conn: sqlite3.Connection, deployment_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, strategy_id, ticker, timeframe, params_json,
               deployment_capital, cost_config_json, order_type,
               schedule_cron, armed, created_at, last_run_at, max_position_value,
               max_loss_value
        FROM deployments WHERE id = ?
        """,
        (deployment_id,),
    ).fetchone()
    return dict(row) if row else None


def list_deployments(
    conn: sqlite3.Connection,
    *,
    strategy_id: str | None = None,
    armed_only: bool = False,
) -> list[dict[str, Any]]:
    """List deployments. Filters are AND-ed."""
    sql = (
        "SELECT id, strategy_id, ticker, timeframe, params_json, "
        "deployment_capital, cost_config_json, order_type, schedule_cron, "
        "armed, created_at, last_run_at, max_position_value, max_loss_value "
        "FROM deployments WHERE 1=1"
    )
    args: list[Any] = []
    if strategy_id is not None:
        sql += " AND strategy_id = ?"
        args.append(strategy_id)
    if armed_only:
        sql += " AND armed = 1"
    sql += " ORDER BY created_at DESC"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def set_deployment_armed(conn: sqlite3.Connection, deployment_id: str, armed: bool) -> bool:
    """Arm or disarm a deployment. Returns True if the row existed."""
    cur = conn.execute(
        "UPDATE deployments SET armed = ? WHERE id = ?",
        (1 if armed else 0, deployment_id),
    )
    return cur.rowcount > 0


def update_deployment_last_run(
    conn: sqlite3.Connection, deployment_id: str, last_run_at: str,
) -> None:
    conn.execute(
        "UPDATE deployments SET last_run_at = ? WHERE id = ?",
        (last_run_at, deployment_id),
    )


def delete_deployment(conn: sqlite3.Connection, deployment_id: str) -> bool:
    """Delete a deployment. ON DELETE CASCADE removes its `orders_sent`
    rows (and, in turn, their fills) and `deployment_runs` history.
    Returns True if the row existed."""
    cur = conn.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))
    return cur.rowcount > 0


def insert_deployment_run(
    conn: sqlite3.Connection,
    *,
    deployment_id: str,
    ran_at: str,
    status: str,
    as_of_timestamp: str | None,
    intent_count: int,
    submitted_count: int,
    error: str | None,
) -> None:
    """Phase 14 — append one row to deployment_runs after every fire of
    run_deployment so the Tracking UI can show a real audit trail."""
    conn.execute(
        """
        INSERT INTO deployment_runs (
            deployment_id, ran_at, status, as_of_timestamp,
            intent_count, submitted_count, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (deployment_id, ran_at, status, as_of_timestamp,
         intent_count, submitted_count, error),
    )


def list_deployment_runs(
    conn: sqlite3.Connection, deployment_id: str, *, limit: int = 50,
) -> list[dict[str, Any]]:
    """Most-recent-first history of a deployment's fires (Phase 14)."""
    rows = conn.execute(
        """
        SELECT id, deployment_id, ran_at, status, as_of_timestamp,
               intent_count, submitted_count, error
        FROM deployment_runs
        WHERE deployment_id = ?
        ORDER BY ran_at DESC, id DESC
        LIMIT ?
        """,
        (deployment_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def list_deployment_runs_since(
    conn: sqlite3.Connection, *, since_id: int = 0, limit: int = 50,
) -> list[dict[str, Any]]:
    """Phase 14 — cross-deployment notifier feed: every run row with
    id > since_id, latest first, joined with the deployment's ticker so
    the UI can display "AAPL: ran" without a second lookup. The frontend
    polls this with the highest id it has already shown so the response
    only carries fresh rows."""
    rows = conn.execute(
        """
        SELECT r.id, r.deployment_id, r.ran_at, r.status, r.as_of_timestamp,
               r.intent_count, r.submitted_count, r.error,
               d.ticker
        FROM deployment_runs r
        JOIN deployments d ON d.id = r.deployment_id
        WHERE r.id > ?
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (since_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def max_deployment_run_id(conn: sqlite3.Connection) -> int:
    """Current max id in deployment_runs, or 0 if empty. The frontend
    uses this on first poll to set the baseline — only fires that
    happen during the session generate notifications, not a backlog."""
    row = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS m FROM deployment_runs"
    ).fetchone()
    return int(row["m"] if row else 0)


def insert_order(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    deployment_id: str,
    request_id_hash: str,
    as_of_timestamp: str,
    ticker: str,
    side: str,
    qty: float,
    order_type: str,
    limit_price: float | None,
    status: str,
    created_at: str,
) -> None:
    """Persist a new order intent. Raises sqlite3.IntegrityError if
    request_id_hash already exists — callers MUST catch that and treat
    it as "already submitted; skip." That is the idempotency primitive
    for the entire deploy pipeline."""
    conn.execute(
        """
        INSERT INTO orders_sent (
            id, deployment_id, request_id_hash, as_of_timestamp, ticker,
            side, qty, order_type, limit_price, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id, deployment_id, request_id_hash, as_of_timestamp, ticker,
            side, qty, order_type, limit_price, status, created_at,
        ),
    )


def find_order_by_request_id(
    conn: sqlite3.Connection, request_id_hash: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, deployment_id, request_id_hash, t212_order_id,
               as_of_timestamp, ticker, side, qty, order_type, limit_price,
               status, response_json, created_at
        FROM orders_sent WHERE request_id_hash = ?
        """,
        (request_id_hash,),
    ).fetchone()
    return dict(row) if row else None


def update_order_status(
    conn: sqlite3.Connection,
    order_id: str,
    *,
    status: str,
    t212_order_id: int | None = None,
    response_json: str | None = None,
) -> bool:
    """Update the local status (and optionally the T212 order id +
    response blob) after submission. Returns True if the row existed."""
    sets = ["status = ?"]
    args: list[Any] = [status]
    if t212_order_id is not None:
        sets.append("t212_order_id = ?")
        args.append(t212_order_id)
    if response_json is not None:
        sets.append("response_json = ?")
        args.append(response_json)
    args.append(order_id)
    sql = f"UPDATE orders_sent SET {', '.join(sets)} WHERE id = ?"
    return conn.execute(sql, args).rowcount > 0


def list_orders_for_deployment(
    conn: sqlite3.Connection, deployment_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, deployment_id, request_id_hash, t212_order_id,
               as_of_timestamp, ticker, side, qty, order_type, limit_price,
               status, response_json, created_at
        FROM orders_sent WHERE deployment_id = ? ORDER BY created_at DESC
        """,
        (deployment_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_fill(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    t212_order_id: int | None,
    filled_qty: float,
    filled_price: float,
    filled_at: str,
    raw_json: str,
) -> int:
    """Persist a fill row. Returns the autoincrement id."""
    cur = conn.execute(
        """
        INSERT INTO fills_received (
            order_id, t212_order_id, filled_qty, filled_price, filled_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (order_id, t212_order_id, filled_qty, filled_price, filled_at, raw_json),
    )
    return int(cur.lastrowid or 0)


def list_fills_for_order(conn: sqlite3.Connection, order_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, order_id, t212_order_id, filled_qty, filled_price, filled_at, raw_json
        FROM fills_received WHERE order_id = ? ORDER BY filled_at
        """,
        (order_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# -------- end deployments --------


def list_runs_with_metrics(
    conn: sqlite3.Connection,
    *,
    strategy_id: str | None = None,
    ticker: str | None = None,
    latest_per_config: bool = False,
) -> list[dict[str, Any]]:
    """All runs joined with their strategy name and metrics. Filters are AND-ed.

    The metrics_json is parsed for the caller; runs without a metrics row are
    skipped (shouldn't happen normally — every successful run inserts metrics
    in the same transaction). Returned rows include the run's `params_json`
    so callers can distinguish single / sweep / walk-forward modes via the
    embedded `mode` key.

    When `latest_per_config` is set, runs are de-duplicated by their full
    config — (strategy_id, ticker, timeframe, start_date, end_date,
    params_json) — keeping only the most recent run for each config. Rows
    arrive `created_at DESC`, so the first row seen for a config is the
    latest. De-dup is display-only: no runs are deleted from the database.
    """
    where_parts = ["s.active = 1"]
    params: list[Any] = []
    if strategy_id is not None:
        where_parts.append("r.strategy_id = ?")
        params.append(strategy_id)
    if ticker is not None:
        where_parts.append("r.ticker = ?")
        params.append(ticker)
    where_sql = " AND ".join(where_parts)
    rows = conn.execute(
        f"""
        SELECT r.id AS run_id, r.strategy_id, s.name AS strategy_name,
               s.favourite AS favourite,
               r.ticker, r.timeframe, r.start_date, r.end_date,
               r.params_json, r.created_at,
               m.metrics_json
        FROM runs r
        JOIN strategies s ON s.id = r.strategy_id
        LEFT JOIN metrics m ON m.run_id = r.id
        WHERE {where_sql}
        ORDER BY r.created_at DESC
        """,
        params,
    ).fetchall()
    out: list[dict[str, Any]] = []
    seen_configs: set[tuple[Any, ...]] = set()
    for r in rows:
        if r["metrics_json"] is None:
            continue
        if latest_per_config:
            config_key = (
                r["strategy_id"], r["ticker"], r["timeframe"],
                r["start_date"], r["end_date"], r["params_json"],
            )
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)
        params_dict = json.loads(r["params_json"]) if r["params_json"] else {}
        mode = "single"
        if isinstance(params_dict, dict) and "mode" in params_dict:
            mode = params_dict["mode"]
        out.append({
            "run_id": r["run_id"],
            "strategy_id": r["strategy_id"],
            "strategy_name": r["strategy_name"],
            "favourite": bool(r["favourite"]),
            "ticker": r["ticker"],
            "timeframe": r["timeframe"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "mode": mode,
            "created_at": r["created_at"],
            "metrics": json.loads(r["metrics_json"]),
        })
    return out


def count_distinct_tickers_per_strategy(conn: sqlite3.Connection) -> dict[str, int]:
    """{strategy_id: distinct_ticker_count}. Used for breadth scoring."""
    rows = conn.execute(
        "SELECT strategy_id, COUNT(DISTINCT ticker) AS n FROM runs GROUP BY strategy_id",
    ).fetchall()
    return {r["strategy_id"]: r["n"] for r in rows}
