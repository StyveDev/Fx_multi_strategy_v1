"""
Centralized logging.

Every module gets its logger via `get_logger(__name__)` — all logs flow into
one rotating file (so the dashboard has a single source to tail) plus the
console. Nothing else in the system should call logging.basicConfig itself.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_CONFIGURED = False


def configure_logging(log_dir: str = "logs", log_file: str = "system.log", level: str = "INFO"):
    global _CONFIGURED
    if _CONFIGURED:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
