"""
Filesystem paths for persistent state outside the project repo.

Tests monkeypatch DB_PATH, DB_DIR, and STRATEGIES_DIR to redirect to tmp_path.
"""
from __future__ import annotations

from pathlib import Path

DATA_ROOT: Path = Path.home() / ".backtester"
DB_DIR: Path = DATA_ROOT / "results"
DB_PATH: Path = DB_DIR / "results.db"
STRATEGIES_DIR: Path = DATA_ROOT / "strategies"


def ensure_dirs() -> None:
    """Create the storage directories if they don't already exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
