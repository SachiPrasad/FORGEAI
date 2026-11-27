"""Structured file + console logging utility for FORGEAI."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def get_forgeai_logger(name: str = "forgeai", log_dir: str | Path = "logs") -> logging.Logger:
    """Return a logger writing Rich output to console and run logs to `<log_dir>/forgeai.log`."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        console_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    try:
        log_root = Path(log_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        target_log_file = (log_root / "forgeai.log").resolve()

        # Replace any FileHandler pointing to a different run directory
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                if Path(handler.baseFilename).resolve() == target_log_file:
                    return logger
                logger.removeHandler(handler)
                handler.close()

        file_handler = logging.FileHandler(target_log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)
    except OSError:
        pass

    return logger
