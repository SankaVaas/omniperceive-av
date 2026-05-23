"""
Logger — structured console + file logging.
Rich-formatted console output with colour-coded severity.
Falls back gracefully if rich is not installed.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "runs/", level: int = logging.INFO) -> logging.Logger:
    """
    Build a logger that writes to stdout (coloured via rich if available)
    and to a timestamped file under log_dir.

    Args:
        name    : logger name (usually the experiment name).
        log_dir : directory to write the log file.
        level   : logging level (default INFO).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-call
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────
    try:
        from rich.logging import RichHandler
        console = RichHandler(rich_tracebacks=True, markup=True)
        console.setLevel(level)
        logger.addHandler(console)
    except ImportError:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    # ── File handler ──────────────────────────────────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{name}_{ts}.log"
    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    logger.info(f"Logger initialised → {log_path}")
    return logger