"""
Cache compaction maintenance (Phase 14).

Rewrites every Parquet file in ~/.backtester/prices/ via the data layer's
own read/write helpers — which sort by timestamp, dedupe duplicate rows,
and re-apply pyarrow compression. Files that were already optimal end up
the same size; fragmented ones shrink. Then runs SQLite VACUUM on
results.db to reclaim space from deleted rows.

Both operations are safe and idempotent. Writes are atomic (temp file +
rename) so a crash mid-run never leaves a half-written Parquet.

Run:
    python scripts/compact_cache.py           # report + compact
    python scripts/compact_cache.py --dry-run # report only, no changes
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Ensure the project root is importable when running as `python scripts/foo.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.data.prices import CACHE_ROOT, _read_cache, _write_cache_atomic
from backend.storage import DB_PATH


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _iter_cache_files() -> list[tuple[str, str, Path]]:
    """Yield (timeframe, ticker, path) for every cached Parquet."""
    if not CACHE_ROOT.exists():
        return []
    out: list[tuple[str, str, Path]] = []
    for tf_dir in sorted(CACHE_ROOT.iterdir()):
        if not tf_dir.is_dir():
            continue
        for f in sorted(tf_dir.glob("*.parquet")):
            out.append((tf_dir.name, f.stem, f))
    return out


def compact_parquet(dry_run: bool) -> tuple[int, int, int]:
    """Rewrite every cached Parquet via the data layer's read/write helpers.

    Returns (files, bytes_before, bytes_after).
    """
    files = _iter_cache_files()
    if not files:
        print("Price cache is empty — nothing to compact.")
        return (0, 0, 0)

    print(f"Found {len(files)} Parquet files under {CACHE_ROOT}.")
    bytes_before = sum(p.stat().st_size for _, _, p in files)
    bytes_after = 0

    for timeframe, ticker, path in files:
        before = path.stat().st_size
        if dry_run:
            print(f"  [dry-run] {timeframe}/{ticker}  ({_fmt_bytes(before)})")
            bytes_after += before
            continue
        try:
            df = _read_cache(ticker, timeframe)
        except Exception as e:  # noqa: BLE001 — report and skip; don't corrupt
            print(f"  ! {timeframe}/{ticker}: read failed ({e!s}); skipping")
            bytes_after += before
            continue
        if df.empty:
            print(f"  - {timeframe}/{ticker}: empty cache; skipping")
            bytes_after += before
            continue
        # Defensive: sort + dedupe in case the on-disk file isn't already.
        df = df[~df.index.duplicated(keep="last")].sort_index()
        _write_cache_atomic(df, ticker, timeframe)
        after = path.stat().st_size
        bytes_after += after
        delta = after - before
        arrow = "→" if delta == 0 else ("↓" if delta < 0 else "↑")
        print(f"  {arrow} {timeframe}/{ticker}  "
              f"{_fmt_bytes(before)} → {_fmt_bytes(after)}  ({len(df)} rows)")

    return (len(files), bytes_before, bytes_after)


def vacuum_results_db(dry_run: bool) -> None:
    """Run SQLite VACUUM on the results database to reclaim deleted-row space."""
    if not DB_PATH.exists():
        print(f"Results DB not found at {DB_PATH} — skipping VACUUM.")
        return
    before = DB_PATH.stat().st_size
    print(f"\nresults.db: {_fmt_bytes(before)} before VACUUM.")
    if dry_run:
        print("  [dry-run] would VACUUM.")
        return
    # VACUUM cannot run inside a transaction — sqlite3 opens in autocommit
    # mode when isolation_level=None.
    with sqlite3.connect(str(DB_PATH), isolation_level=None) as conn:
        conn.execute("VACUUM")
    after = DB_PATH.stat().st_size
    print(f"  VACUUM done. {_fmt_bytes(before)} → {_fmt_bytes(after)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report only — do not rewrite Parquet files or VACUUM the DB",
    )
    args = parser.parse_args()

    print(f"Cache compaction {'(dry run)' if args.dry_run else ''}".rstrip())
    print("-" * 60)
    files, before, after = compact_parquet(args.dry_run)
    if files:
        delta = before - after
        print(
            f"\nParquet total: {_fmt_bytes(before)} → {_fmt_bytes(after)}  "
            f"(reclaimed {_fmt_bytes(max(delta, 0))})"
        )

    vacuum_results_db(args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
