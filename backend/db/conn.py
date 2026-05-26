"""
SQLite connection management + schema migration.

Per-connection PRAGMAs:
  - foreign_keys=ON (so ON DELETE CASCADE actually works)
  - journal_mode=WAL (better concurrent reads, durable writes)
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_PATH: Path = Path(__file__).parent / "schema.sql"


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Yield a configured SQLite connection. Commits on success, rolls back on exception."""
    # Read paths dynamically inside the function so that monkeypatching
    # backend.storage.DB_PATH/DB_DIR in tests takes effect on every call.
    from backend.storage import DB_DIR, DB_PATH

    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        _configure(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added to a table after its initial CREATE. `migrate()` ALTERs
# them into a pre-existing DB that predates them (CREATE TABLE IF NOT
# EXISTS will not add a column to a table that already exists). schema.sql
# also carries every column, so a fresh DB gets them on creation.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "deployments": {
        "max_position_value": "REAL",  # Phase 12.3
        "max_loss_value": "REAL",      # Phase 12.4
    },
    "strategies": {
        "favourite": "INTEGER NOT NULL DEFAULT 0",  # Phase 15.4
    },
    "runs": {
        "oos_json": "TEXT",            # Phase 16.1
        "dsr_json": "TEXT",            # Phase 16.1
        "monte_carlo_json": "TEXT",    # Phase 16.1
        "strategy_code_hash": "TEXT",  # Phase 16.2
        "notes": "TEXT",               # Phase 16.3
        "tags": "TEXT",                # Phase 16.3
    },
}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """ALTER-add any column in `_ADDED_COLUMNS` missing from its table."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _backfill_strategy_versions(conn: sqlite3.Connection) -> None:
    """Phase 16.2 — synthesize one strategy_versions row per active
    strategy that doesn't have one yet (e.g. strategies created before
    16.2 shipped). Reads each strategy's current code from disk and
    inserts a version row using its content hash. Idempotent thanks to
    the UNIQUE (strategy_id, code_hash) constraint.
    """
    import hashlib
    from datetime import datetime, timezone
    from pathlib import Path

    missing = conn.execute(
        """
        SELECT s.id, s.code_path, s.created_at
        FROM strategies s
        LEFT JOIN strategy_versions v ON v.strategy_id = s.id
        WHERE s.active = 1 AND v.id IS NULL
        GROUP BY s.id
        """
    ).fetchall()
    for row in missing:
        path = Path(row["code_path"])
        if not path.exists():
            continue
        code = path.read_text(encoding="utf-8")
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        # Use the strategy's own created_at so the synthetic version
        # carries the right timestamp instead of "right now".
        created_at = row["created_at"] or datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO strategy_versions "
            "(strategy_id, code_hash, code, created_at) VALUES (?, ?, ?, ?)",
            (row["id"], code_hash, code, created_at),
        )


def migrate() -> None:
    """Apply schema.sql, then ALTER-in any columns added to existing tables
    after their initial CREATE. Idempotent."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema_sql)
        _ensure_columns(conn)
        _backfill_strategy_versions(conn)
