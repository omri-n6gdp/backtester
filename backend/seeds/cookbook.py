"""
Phase 16.5 — first-boot cookbook seeder.

Loads `seeds/cookbook/manifest.json` and inserts each listed strategy
into the database the first time the server starts. Idempotent: keyed
on the strategy's name (e.g. "Cookbook · SMA crossover"); if a strategy
with that exact name already exists, the entry is skipped.

Called from `db.conn.migrate()` after the version backfill so a fresh
install gets six worked examples without manual setup.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.db import repo

logger = logging.getLogger(__name__)

# `seeds/` lives at the repo root, next to `backend/`. Walk two parents
# up from this file (backend/seeds/cookbook.py) and pick "seeds/cookbook".
_SEEDS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "seeds" / "cookbook"


def seed_cookbook(conn: sqlite3.Connection) -> int:
    """Insert any cookbook strategies that aren't already present.
    Returns the number of strategies newly inserted (0 on subsequent
    startups). Validation failures are logged and skipped so a bad
    seed file can't block the server."""
    manifest_path = _SEEDS_DIR / "manifest.json"
    if not manifest_path.exists():
        return 0

    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("cookbook seeder: manifest.json parse failed: %s", e)
        return 0

    # Names already present — skip those.
    existing = {
        row["name"]
        for row in conn.execute("SELECT name FROM strategies WHERE active = 1")
    }

    # Late import — the validator module pulls in the engine; importing
    # at the top of this file would create a circular import with
    # backend.db.conn (which imports us).
    from backend.engine.validator import validate_strategy
    from backend.storage import STRATEGIES_DIR

    inserted = 0
    for entry in entries:
        name = entry.get("name")
        rel_file = entry.get("file")
        description = entry.get("description", "")
        if not name or not rel_file:
            logger.warning("cookbook seeder: bad manifest entry %s", entry)
            continue
        if name in existing:
            continue

        src = _SEEDS_DIR / rel_file
        if not src.exists():
            logger.warning("cookbook seeder: missing source %s", src)
            continue
        code = src.read_text(encoding="utf-8")

        vresult = validate_strategy(code)
        if not vresult.ok:
            logger.error(
                "cookbook seeder: %s failed validation — %s",
                name, "; ".join(vresult.errors),
            )
            continue

        strategy_id = uuid.uuid4().hex
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
        (STRATEGIES_DIR / f"{strategy_id}.py").write_text(code, encoding="utf-8")

        created_at = datetime.now(timezone.utc).isoformat()
        repo.insert_strategy(
            conn,
            strategy_id=strategy_id,
            name=name,
            description=description,
            code_path=str(STRATEGIES_DIR / f"{strategy_id}.py"),
            created_at=created_at,
        )
        repo.insert_strategy_version(
            conn, strategy_id=strategy_id, code=code, created_at=created_at,
        )
        inserted += 1
        logger.info("cookbook seeder: inserted %s", name)

    return inserted
