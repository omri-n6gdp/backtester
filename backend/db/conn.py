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


def migrate() -> None:
    """Apply schema.sql. Idempotent — uses CREATE TABLE IF NOT EXISTS throughout."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema_sql)
