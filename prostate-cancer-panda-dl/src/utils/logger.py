"""Logging utilities for scripts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .io import ensure_dir


def setup_logger(name: str, logs_dir: Path) -> logging.Logger:
    """Create a script logger with file + console handlers."""
    ensure_dir(logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Log file: %s", log_path)
    return logger

