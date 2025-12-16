"""
Logging configuration for AgorAI package.

This module sets up structured logging to files with rotation and proper formatting.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

from .config import config


class AgorAIFormatter(logging.Formatter):
    """Custom formatter with structured logging format."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with timestamp, level, module, and message."""
        # Add timestamp
        record.timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Format: [TIMESTAMP] [LEVEL] [module:function:line] MESSAGE
        return super().format(record)


def setup_logging(name: str = "agorai", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for the AgorAI package.

    Args:
        name: Logger name (default: "agorai")
        log_file: Optional log file name (default: agorai_YYYYMMDD.log)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Set log level from config
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist
    log_dir = config.get_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    # Create log file name with date
    if log_file is None:
        log_file = f"agorai_{datetime.now().strftime('%Y%m%d')}.log"

    log_path = os.path.join(log_dir, log_file)

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)

    # Create formatter
    formatter = AgorAIFormatter(
        fmt='[%(timestamp)s] [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid console output)
    logger.propagate = False

    # Log initial message
    logger.info(f"AgorAI logging initialized - Log file: {log_path}")

    return logger


def get_logger(name: str = "agorai") -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (default: "agorai")

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger not yet configured, set it up
    if not logger.handlers:
        return setup_logging(name)

    return logger
