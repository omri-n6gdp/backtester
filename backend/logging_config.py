"""
Logging configuration (Phase 14).

Modules across the backend already call ``logging.getLogger(__name__)``
and emit info/warning/exception records — but with no configured root
logger those records go nowhere (Python's default only shows WARNING+
on stderr). ``setup_logging()`` wires the root logger to a rotating file
in ``~/.backtester/logs/`` plus the console, so a server boot lights up
the log calls that already exist throughout the codebase.

Called once from ``server.py``; never from the test suite (which would
not want handlers attached to the shared root logger).
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from backend.storage import LOGS_DIR

LOG_FILE = LOGS_DIR / "backtester.log"
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

# Library loggers that are chatty at INFO — kept at WARNING so the log
# trail stays about our own activity, not every HTTP request yfinance makes.
_NOISY_LIBRARIES = ("httpx", "httpcore", "yfinance", "apscheduler", "urllib3")

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger: a rotating file handler in
    ``~/.backtester/logs/`` (2 MB per file, 5 backups) plus a console
    handler. Idempotent — safe to call more than once."""
    global _configured
    if _configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for noisy in _NOISY_LIBRARIES:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info("logging configured -> %s", LOG_FILE)
