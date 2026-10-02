"""
Structured logging setup for the AI Training Supervisor.
Creates two handlers:
  1. A rotating file handler for general application logs.
  2. A dedicated audit-log handler that records every action/decision.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Union


AUDIT_LOGGER = "ats.audit"


def setup_logging(
    log_level: str = "INFO",
    log_dir: Union[Path, str] = "logs",
    audit_log: Union[Path, str] = "logs/audit.log",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure root and audit loggers.

    Parameters
    ----------
    log_level:
        Root logging level (DEBUG / INFO / WARNING / ERROR).
    log_dir:
        Directory where rotating application logs are stored.
    audit_log:
        Path to the append-only audit log file.
    max_bytes:
        Maximum size of each rotating log file.
    backup_count:
        Number of backup log files to keep.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    audit_log_path = Path(audit_log)
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Root logger ────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # Rotating file handler
    app_log = log_dir / "supervisor.log"
    file_handler = logging.handlers.RotatingFileHandler(
        app_log, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # ── Audit logger (append-only, always INFO+) ───────────────────
    audit_logger = logging.getLogger(AUDIT_LOGGER)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False          # Don't duplicate to root

    audit_handler = logging.handlers.RotatingFileHandler(
        audit_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | AUDIT | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    audit_logger.addHandler(audit_handler)


def get_audit_logger() -> logging.Logger:
    """Return the dedicated audit logger."""
    return logging.getLogger(AUDIT_LOGGER)
