"""
Logging Configuration
=====================
Sets up structured logging with console and rotating file handlers.
Import and call ``setup_logging()`` once at application startup.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from config.settings import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_FILE,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOGS_DIR,
)


def setup_logging(name: str | None = None) -> logging.Logger:
    """
    Configure and return the application logger.

    Parameters
    ----------
    name : str, optional
        Logger name.  Defaults to the root logger when ``None``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Ensure log directory exists
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Console handler ──────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Rotating file handler ────────────────────────
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(LOG_FILE),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        # Graceful degradation: log only to console if file is not writable
        logger.warning("Could not create log file at %s — logging to console only.", LOG_FILE)

    return logger
