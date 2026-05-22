"""
Tests for backend/logging_config.py (Phase 14).

setup_logging() mutates the shared root logger, so each test snapshots
and restores its handlers to avoid leaking them into the rest of the run.
"""
from __future__ import annotations

import logging

from backend import logging_config


def test_setup_logging_creates_file_and_is_idempotent(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(logging_config, "LOGS_DIR", log_dir)
    monkeypatch.setattr(logging_config, "LOG_FILE", log_dir / "backtester.log")
    monkeypatch.setattr(logging_config, "_configured", False)

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        logging_config.setup_logging()
        logging.getLogger("test.logging").info("hello from the test")

        assert (log_dir / "backtester.log").exists()
        n_handlers = len(root.handlers)

        # Second call is a no-op — no duplicate handlers.
        logging_config.setup_logging()
        assert len(root.handlers) == n_handlers
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_noisy_libraries_pinned_to_warning(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(logging_config, "LOGS_DIR", log_dir)
    monkeypatch.setattr(logging_config, "LOG_FILE", log_dir / "backtester.log")
    monkeypatch.setattr(logging_config, "_configured", False)

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        logging_config.setup_logging()
        # yfinance's chatter is pinned above INFO so the log stays useful.
        assert logging.getLogger("yfinance").level == logging.WARNING
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
