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
    if include_inactive:
        rows = conn.execute(
            "SELECT id, name, description, code_path, created_at, active "
            "FROM strategies ORDER BY created_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, description, code_path, created_at, active "
            "FROM strategies WHERE active = 1 ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_strategy(conn: sqlite3.Connection, strategy_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, name, description, code_path, created_at, active "
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


def list_runs_with_metrics(
    conn: sqlite3.Connection,
    *,
    strategy_id: str | None = None,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """All runs joined with their strategy name and metrics. Filters are AND-ed.

    The metrics_json is parsed for the caller; runs without a metrics row are
    skipped (shouldn't happen normally — every successful run inserts metrics
    in the same transaction). Returned rows include the run's `params_json`
    so callers can distinguish single / sweep / walk-forward modes via the
    embedded `mode` key.
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
    for r in rows:
        if r["metrics_json"] is None:
            continue
        params_dict = json.loads(r["params_json"]) if r["params_json"] else {}
        mode = "single"
        if isinstance(params_dict, dict) and "mode" in params_dict:
            mode = params_dict["mode"]
        out.append({
            "run_id": r["run_id"],
            "strategy_id": r["strategy_id"],
            "strategy_name": r["strategy_name"],
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
