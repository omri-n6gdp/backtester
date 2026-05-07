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
